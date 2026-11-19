"""Validation-set decision threshold calibration for Aegis Forensics.

Evaluates P(FAKE) >= threshold across a sweep of thresholds strictly on Dataset/valid
(never on Dataset/test, avoiding test leakage) without modifying raw softmax
probabilities. Helps trade off FAKE Recall (reducing fake-missed-as-real FNR) against
REAL False Positive Rate (FPR) cleanly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exif_rgb_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def calibrate(
    model_path: Path,
    max_fpr: float = 0.10,
    sample_size: int | None = None,
    seed: int = 42,
) -> dict:
    valid_dir = DATASET_PATH / "valid"
    if not valid_dir.exists():
        raise FileNotFoundError(f"Validation dataset folder not found: {valid_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    full_valid = datasets.ImageFolder(valid_dir, transform=transform, loader=exif_rgb_loader)
    if full_valid.class_to_idx != {"fake": 0, "real": 1}:
        raise ValueError(
            f"Unexpected class mapping: {full_valid.class_to_idx}. Expected {{'fake': 0, 'real': 1}}."
        )

    if sample_size is None:
        selected_indices = list(range(len(full_valid)))
    else:
        fake_indices = [i for i, (_, y) in enumerate(full_valid.samples) if y == 0]
        real_indices = [i for i, (_, y) in enumerate(full_valid.samples) if y == 1]
        per_class = min(sample_size // 2, len(fake_indices), len(real_indices))
        rng = random.Random(seed)
        selected_indices = rng.sample(fake_indices, per_class) + rng.sample(real_indices, per_class)
        rng.shuffle(selected_indices)

    loader = DataLoader(Subset(full_valid, selected_indices), batch_size=32, shuffle=False, num_workers=0)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if checkpoint.get("class_to_idx") != {"fake": 0, "real": 1}:
        raise ValueError(f"Checkpoint class_to_idx mismatch: {checkpoint.get('class_to_idx')}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    y_true: list[int] = []
    fake_probs: list[float] = []

    print(f"Running validation inference on {len(selected_indices)} images from {valid_dir} using {model_path.name}...")
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(labels.tolist())
            fake_probs.extend(probs[:, 0].cpu().tolist())

    thresholds = [round(0.25 + 0.025 * i, 3) for i in range(21)]  # 0.250 .. 0.750
    sweep: list[dict] = []

    print("\nVALIDATION THRESHOLD SWEEP (predict FAKE if P(FAKE) >= t)")
    print("=" * 88)
    print(
        f"{'t':>6} {'Acc':>8} {'FakePrec':>9} {'FakeRec':>9} {'FNR':>8} "
        f"{'RealRec':>9} {'FPR':>8} {'MacroF1':>9} {'FakeF2':>8}"
    )
    print("-" * 88)

    for t in thresholds:
        y_pred = [
            0 if (pf > 1.0 - pf if abs(t - 0.5) < 1e-9 else pf >= t) else 1
            for pf in fake_probs
        ]
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        macro_f1 = float((f1[0] + f1[1]) / 2.0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp_fake, fn_fake = int(cm[0, 0]), int(cm[0, 1])
        fp_real, tn_real = int(cm[1, 0]), int(cm[1, 1])
        fnr = fn_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) else 0.0
        fpr = fp_real / (fp_real + tn_real) if (fp_real + tn_real) else 0.0
        fake_prec = float(prec[0])
        fake_rec = float(rec[0])
        denom = 4.0 * fake_prec + fake_rec
        fake_f2 = (5.0 * fake_prec * fake_rec / denom) if denom > 0 else 0.0

        row = {
            "threshold": t,
            "accuracy": round(acc, 6),
            "fake_precision": round(fake_prec, 6),
            "fake_recall": round(fake_rec, 6),
            "fnr": round(fnr, 6),
            "real_precision": round(float(prec[1]), 6),
            "real_recall": round(float(rec[1]), 6),
            "fpr": round(fpr, 6),
            "macro_f1": round(macro_f1, 6),
            "fake_f2": round(fake_f2, 6),
        }
        sweep.append(row)
        print(
            f"{t:>6.3f} {acc:>7.2%} {fake_prec:>8.2%} {fake_rec:>8.2%} {fnr:>7.2%} "
            f"{float(rec[1]):>8.2%} {fpr:>7.2%} {macro_f1:>9.4f} {fake_f2:>8.4f}"
        )

    best_macro_f1 = max(sweep, key=lambda r: (r["macro_f1"], r["fake_recall"]))
    constrained = [r for r in sweep if r["fpr"] <= max_fpr]
    best_recall_under_fpr = (
        max(constrained, key=lambda r: (r["fake_recall"], -r["fpr"]))
        if constrained
        else min(sweep, key=lambda r: r["fpr"])
    )

    print("\nRECOMMENDED OPERATING POINTS (CALIBRATED ON Dataset/valid)")
    print("=" * 88)
    print(
        f"  1. Best Macro F1:            t={best_macro_f1['threshold']:.3f} | "
        f"FakeRec={best_macro_f1['fake_recall']:.2%} (FNR={best_macro_f1['fnr']:.2%}) | "
        f"RealRec={best_macro_f1['real_recall']:.2%} (FPR={best_macro_f1['fpr']:.2%}) | "
        f"MacroF1={best_macro_f1['macro_f1']:.4f}"
    )
    print(
        f"  2. Max Fake Recall (FPR<={max_fpr:.0%}): t={best_recall_under_fpr['threshold']:.3f} | "
        f"FakeRec={best_recall_under_fpr['fake_recall']:.2%} (FNR={best_recall_under_fpr['fnr']:.2%}) | "
        f"RealRec={best_recall_under_fpr['real_recall']:.2%} (FPR={best_recall_under_fpr['fpr']:.2%}) | "
        f"FakePrec={best_recall_under_fpr['fake_precision']:.2%}"
    )

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": model_path.name,
        "dataset": "Dataset/valid",
        "evaluated_samples": len(y_true),
        "max_fpr_constraint": max_fpr,
        "best_macro_f1": best_macro_f1,
        "recommended_recall_oriented": best_recall_under_fpr,
        "sweep": sweep,
    }
    out_path = RESULTS_DIR / "threshold_calibration.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    rec_t = best_recall_under_fpr["threshold"]
    print(f"\nSaved calibration table to: {out_path}")
    print(
        "\nNext steps:\n"
        f"  1. Verify calibrated threshold on untouched Dataset/test:\n"
        f"     python scripts/evaluate_model.py --model-path {model_path} --decision-threshold {rec_t}\n"
        f"  2. Run FastAPI with this calibrated threshold (without changing softmax probs):\n"
        f"     FAKE_DECISION_THRESHOLD={rec_t} MODEL_PATH={model_path} uvicorn backend.main:app --host 0.0.0.0 --port 8000"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FAKE decision threshold on Dataset/valid.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.10,
        help="Maximum acceptable REAL false-positive rate when selecting recall-oriented threshold (default: 0.10).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional balanced validation sample size (default: None = full Dataset/valid).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    calibrate(
        model_path=args.model_path,
        max_fpr=args.max_fpr,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
