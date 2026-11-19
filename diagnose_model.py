"""Diagnostic script for Aegis Forensics EfficientNet-B0 checkpoint & pipeline.

Does NOT retrain or modify models/best_model.pth.
Run from project root:
    python diagnose_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import urllib.request

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pth"
DEFAULT_TEST_DIR = ROOT / "Dataset" / "test"

EXPECTED_CLASS_MAPPING = {"fake": 0, "real": 1}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_training_parity_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Exact validation/inference preprocessing matching train_model_d6a819f4.py."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_binary_metrics(y_true: list[int], y_pred: list[int], real_probs: list[float]) -> dict:
    """Compute confusion matrix, accuracy, precision, recall, F1, and ROC-AUC."""
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)  # fake -> fake
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)  # fake -> real
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)  # real -> fake
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)  # real -> real

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0

    # Per-class precision, recall, F1
    fake_prec = tn / (tn + fn) if (tn + fn) else 0.0
    fake_rec = tn / (tn + fp) if (tn + fp) else 0.0
    fake_f1 = (2 * fake_prec * fake_rec / (fake_prec + fake_rec)) if (fake_prec + fake_rec) else 0.0

    real_prec = tp / (tp + fp) if (tp + fp) else 0.0
    real_rec = tp / (tp + fn) if (tp + fn) else 0.0
    real_f1 = (2 * real_prec * real_rec / (real_prec + real_rec)) if (real_prec + real_rec) else 0.0

    macro_prec = (fake_prec + real_prec) / 2.0
    macro_rec = (fake_rec + real_rec) / 2.0
    macro_f1 = (fake_f1 + real_f1) / 2.0

    roc_auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, real_probs))
    except Exception:
        # Pure-Python Mann-Whitney U AUC fallback
        pos = [p for y, p in zip(y_true, real_probs) if y == 1]
        neg = [p for y, p in zip(y_true, real_probs) if y == 0]
        if pos and neg:
            wins = 0.0
            for p_pos in pos:
                for p_neg in neg:
                    if p_pos > p_neg:
                        wins += 1.0
                    elif p_pos == p_neg:
                        wins += 0.5
            roc_auc = wins / (len(pos) * len(neg))

    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": accuracy,
        "fake": {"precision": fake_prec, "recall": fake_rec, "f1": fake_f1, "support": tn + fp},
        "real": {"precision": real_prec, "recall": real_rec, "f1": real_f1, "support": fn + tp},
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "roc_auc": roc_auc,
    }


def check_preprocessing_consistency(root: Path) -> list[str]:
    """Audit repository training/inference files for preprocessing & class mapping consistency."""
    print("\n" + "=" * 72)
    print("10. TRAINING VS INFERENCE PREPROCESSING CONSISTENCY CHECK")
    print("=" * 72)
    warnings: list[str] = []

    candidates = [
        root / "train_model.py",
        root / "scripts" / "train_model.py",
        root / "evaluate_model.py",
        root / "scripts" / "evaluate_model.py",
        root / "prepare_data.py",
        root / "scripts" / "prepare_data.py",
        root / "backend" / "preprocessing.py",
        root / "backend" / "predict.py",
        root / "app.py",
        root / "scripts" / "streamlit_app.py",
    ]
    found_any = False
    for path in candidates:
        if not path.exists():
            continue
        found_any = True
        text = path.read_text(encoding="utf-8", errors="ignore")
        has_norm = "0.485" in text and "0.229" in text
        has_224 = "224" in text
        has_to_tensor = "ToTensor" in text
        issues = []
        if "prepare_data" in path.name and not has_norm:
            issues.append("missing ImageNet Normalize")
        if ("train" in path.name or "evaluate" in path.name or "preprocessing" in path.name or "app" in path.name):
            if not has_224:
                issues.append("missing 224x224 resize")
            if not has_to_tensor:
                issues.append("missing ToTensor()")
            if not has_norm:
                issues.append("missing ImageNet Normalize")
        status = "OK" if not issues else f"MISMATCH ({', '.join(issues)})"
        print(f"  - {path.relative_to(root)!s:<32} -> {status}")
        if issues:
            warnings.append(f"{path.relative_to(root)}: {', '.join(issues)}")

    if not found_any:
        print("  No local training/backend scripts found to cross-check.")

    print(
        f"  - Inference transform verified: Resize(({IMAGE_SIZE}, {IMAGE_SIZE})) -> "
        f"ToTensor() [0,1] -> Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})"
    )
    return warnings


def compare_with_fastapi(
    api_url: str,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    samples: list[tuple[str, int]],
) -> None:
    """Compare local PyTorch logits/probs directly against FastAPI POST /api/predict."""
    base = api_url.rstrip("/")
    print("\n" + "=" * 72)
    print(f"11. FASTAPI PARITY COMPARISON ({base}/api/predict)")
    print("=" * 72)

    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=10) as resp:
            print("  /api/health:", resp.read().decode())
    except Exception:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
                print("  /health:", resp.read().decode())
        except Exception as exc:
            print(f"  ERROR reaching health endpoint at {base}: {exc}")
            return

    max_diff = 0.0
    for img_path, true_id in samples:
        p = Path(img_path)
        with Image.open(p) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.inference_mode():
            local_logits = model(tensor)[0]
            local_probs = torch.softmax(local_logits, dim=0)
        l_fake = float(local_probs[0].item())
        l_real = float(local_probs[1].item())

        boundary = "----AegisParityBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + p.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{base}/api/predict?threshold=0.65",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            api_json = json.loads(resp.read().decode())

        api_probs = api_json.get("probabilities", {})
        a_fake = float(api_probs.get("fake", -1.0))
        a_real = float(api_probs.get("real", -1.0))
        diff = max(abs(l_fake - a_fake), abs(l_real - a_real))
        max_diff = max(max_diff, diff)
        true_lbl = "FAKE" if true_id == 0 else "REAL"
        print(
            f"  [{true_lbl}] {p.name:<22} | "
            f"Local=({l_fake:.4f}, {l_real:.4f}) vs "
            f"FastAPI=({a_fake:.4f}, {a_real:.4f}) | diff={diff:.4f}"
        )

    if max_diff > 0.01:
        print(
            f"\n  [MISMATCH DETECTED] Max probability diff = {max_diff:.4f}! "
            "Your FastAPI server is NOT running the same checkpoint or preprocessing as diagnose_model.py."
        )
    else:
        print(f"\n  [PARITY VERIFIED] Max probability diff = {max_diff:.6f} (FastAPI matches local checkpoint).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose EfficientNet-B0 checkpoint and predictions.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--per-class", type=int, default=20, help="Samples per class (default: 20).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional FastAPI URL (e.g. http://127.0.0.1:8001) to verify exact parity against local model.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1 & 2. Load checkpoint, print keys & inspect model architecture
    # ------------------------------------------------------------------
    print("=" * 72)
    print("1 & 2. CHECKPOINT INSPECTION & MODEL ARCHITECTURE")
    print("=" * 72)
    print(f"Checkpoint path: {args.model_path}")

    if not args.model_path.exists():
        print(f"ERROR: Checkpoint file not found at '{args.model_path}'.")
        return 1

    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    except Exception as exc:
        print(f"ERROR loading checkpoint with weights_only=True: {exc}")
        return 1

    if isinstance(checkpoint, dict):
        print(f"Checkpoint type: dict | Keys: {list(checkpoint.keys())}")
        for k, v in checkpoint.items():
            if k not in ("model_state_dict", "state_dict", "optimizer_state_dict"):
                print(f"  checkpoint[{k!r}] = {v!r}")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            print("  NOTE: Checkpoint appears to be a raw state_dict (no metadata keys).")
            state_dict = checkpoint
    else:
        print(f"ERROR: Unexpected checkpoint object type: {type(checkpoint)}")
        return 1

    if not state_dict:
        print("ERROR: Could not find 'model_state_dict' in checkpoint.")
        return 1

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    print("\nModel Architecture:")
    print(f"  Backbone:        torchvision.models.efficientnet_b0")
    print(f"  Classifier Head: {model.classifier}")
    print(f"  Total Params:    {sum(p.numel() for p in model.parameters()):,}")

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True), ([], [])
        print("  State dict load: STRICT MATCH (0 missing, 0 unexpected keys)")
    except RuntimeError as exc:
        print(f"  ERROR: Strict state_dict load failed: {exc}")
        return 1

    non_finite = [
        name for name, tensor in model.state_dict().items()
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all()
    ]
    if non_finite:
        print(f"  ERROR: Non-finite (NaN/Inf) tensors found in: {non_finite}")
        return 1

    w = model.classifier[1].weight.detach()
    b = model.classifier[1].bias.detach()
    print(f"  Classifier weight norm -> class 0 (fake): {w[0].norm().item():.4f} | class 1 (real): {w[1].norm().item():.4f}")
    print(f"  Classifier bias        -> class 0 (fake): {b[0].item():.4f} | class 1 (real): {b[1].item():.4f}")
    weight_diff = float((w[0] - w[1]).abs().max().item())
    print(f"  Max |W_fake - W_real|: {weight_diff:.6f}")

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Verify class_to_idx mapping
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("3. CLASS-TO-INDEX MAPPING CHECK")
    print("=" * 72)
    ckpt_mapping = checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None
    print(f"  Expected mapping:   {EXPECTED_CLASS_MAPPING}")
    print(f"  Checkpoint mapping: {ckpt_mapping}")
    if ckpt_mapping is not None and ckpt_mapping != EXPECTED_CLASS_MAPPING:
        print("  CRITICAL WARNING: Checkpoint class_to_idx differs from {'fake': 0, 'real': 1}!")

    if not args.test_dir.exists():
        print(f"\nERROR: Test dataset directory not found: {args.test_dir}")
        return 1

    test_folder = datasets.ImageFolder(args.test_dir)
    print(f"  Dataset/test mapping: {test_folder.class_to_idx}")
    if test_folder.class_to_idx != EXPECTED_CLASS_MAPPING:
        print("  CRITICAL WARNING: Dataset/test ImageFolder class_to_idx != {'fake': 0, 'real': 1}!")
        return 1

    # ------------------------------------------------------------------
    # 4 & 5. Sample 20 known FAKE + 20 known REAL & apply exact preprocessing
    # ------------------------------------------------------------------
    ckpt_img_size = int(checkpoint.get("image_size", IMAGE_SIZE)) if isinstance(checkpoint, dict) else IMAGE_SIZE
    transform = build_training_parity_transform(ckpt_img_size)

    fake_samples = [path for path, label in test_folder.samples if label == 0]
    real_samples = [path for path, label in test_folder.samples if label == 1]

    print(f"\n  Available in {args.test_dir}: fake={len(fake_samples)}, real={len(real_samples)}")
    rng = random.Random(args.seed)
    n_fake = min(args.per_class, len(fake_samples))
    n_real = min(args.per_class, len(real_samples))
    if n_fake == 0 or n_real == 0:
        print("ERROR: Both Dataset/test/fake and Dataset/test/real must contain images.")
        return 1

    selected = [(p, 0) for p in rng.sample(fake_samples, n_fake)] + [
        (p, 1) for p in rng.sample(real_samples, n_real)
    ]

    # Verify normalized tensor range on first sample
    with Image.open(selected[0][0]) as probe_img:
        probe_rgb = ImageOps.exif_transpose(probe_img).convert("RGB")
        probe_tensor = transform(probe_rgb)
    print(
        f"  Sample tensor stats -> shape={tuple(probe_tensor.shape)}, "
        f"min={probe_tensor.min().item():.3f}, max={probe_tensor.max().item():.3f}, "
        f"mean={probe_tensor.mean().item():.3f}, std={probe_tensor.std().item():.3f}"
    )

    # ------------------------------------------------------------------
    # 6. Run inference & print per-image predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"4–6. PER-IMAGE PREDICTIONS ({n_fake} FAKE + {n_real} REAL = {len(selected)} IMAGES)")
    print("=" * 72)
    header = (
        f"{'#':>2}  {'Filename':<24} {'True':<5} {'Pred':<5} "
        f"{'P(FAKE)':>9} {'P(REAL)':>9} {'Logit0':>8} {'Logit1':>8}  Status"
    )
    print(header)
    print("-" * len(header))

    y_true: list[int] = []
    y_pred: list[int] = []
    all_fake_probs: list[float] = []
    all_real_probs: list[float] = []

    fake_group_p_fake: list[float] = []
    fake_group_p_real: list[float] = []
    real_group_p_fake: list[float] = []
    real_group_p_real: list[float] = []

    with torch.inference_mode():
        for idx, (img_path, true_id) in enumerate(selected, start=1):
            with Image.open(img_path) as img:
                rgb = ImageOps.exif_transpose(img).convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to(device)
            logits = model(tensor)[0]
            probs = torch.softmax(logits, dim=0)

            p_fake = float(probs[0].item())
            p_real = float(probs[1].item())
            pred_id = 1 if p_real >= p_fake else 0

            true_name = "FAKE" if true_id == 0 else "REAL"
            pred_name = "FAKE" if pred_id == 0 else "REAL"
            status = "OK" if pred_id == true_id else "WRONG"

            y_true.append(true_id)
            y_pred.append(pred_id)
            all_fake_probs.append(p_fake)
            all_real_probs.append(p_real)

            if true_id == 0:
                fake_group_p_fake.append(p_fake)
                fake_group_p_real.append(p_real)
            else:
                real_group_p_fake.append(p_fake)
                real_group_p_real.append(p_real)

            fname = Path(img_path).name
            if len(fname) > 24:
                fname = fname[:21] + "..."
            print(
                f"{idx:>2}  {fname:<24} {true_name:<5} {pred_name:<5} "
                f"{p_fake:>8.2%} {p_real:>8.2%} "
                f"{logits[0].item():>8.3f} {logits[1].item():>8.3f}  {status}"
            )

    # ------------------------------------------------------------------
    # 7. Accuracy, Precision, Recall, F1, Confusion Matrix
    # ------------------------------------------------------------------
    metrics = compute_binary_metrics(y_true, y_pred, all_real_probs)
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 72)
    print("7. EVALUATION METRICS & CONFUSION MATRIX")
    print("=" * 72)
    print(f"  Accuracy:        {metrics['accuracy']:.2%} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")
    print(f"  Macro Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  Macro F1-Score:  {metrics['macro']['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(
        f"  FAKE (0) -> Prec={metrics['fake']['precision']:.4f} | "
        f"Rec={metrics['fake']['recall']:.4f} | F1={metrics['fake']['f1']:.4f} | n={metrics['fake']['support']}"
    )
    print(
        f"  REAL (1) -> Prec={metrics['real']['precision']:.4f} | "
        f"Rec={metrics['real']['recall']:.4f} | F1={metrics['real']['f1']:.4f} | n={metrics['real']['support']}"
    )
    print("\n  Confusion Matrix (rows = Actual, cols = Predicted):")
    print("                  Pred FAKE (0)   Pred REAL (1)")
    print(f"    Actual FAKE (0)    {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"    Actual REAL (1)    {cm[1][0]:>5}          {cm[1][1]:>5}")

    # ------------------------------------------------------------------
    # 8 & 9. Separate mean probabilities & single-class bias check
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("8 & 9. PROBABILITY SEPARATION & SINGLE-CLASS COLLAPSE CHECK")
    print("=" * 72)
    mean_fake_pfake = statistics.mean(fake_group_p_fake)
    mean_fake_preal = statistics.mean(fake_group_p_real)
    mean_real_pfake = statistics.mean(real_group_p_fake)
    mean_real_preal = statistics.mean(real_group_p_real)
    std_preal_all = statistics.pstdev(all_real_probs)
    mean_conf = statistics.mean(max(pf, pr) for pf, pr in zip(all_fake_probs, all_real_probs))
    near_fifty_count = sum(1 for pr in all_real_probs if 0.45 <= pr <= 0.55)
    low_conf_count = sum(1 for pf, pr in zip(all_fake_probs, all_real_probs) if max(pf, pr) < 0.65)

    pred_fake_total = sum(1 for p in y_pred if p == 0)
    pred_real_total = sum(1 for p in y_pred if p == 1)
    dominant_share = max(pred_fake_total, pred_real_total) / len(y_pred)

    print(f"  Known FAKE images -> avg P(FAKE)={mean_fake_pfake:.4f}, avg P(REAL)={mean_fake_preal:.4f}")
    print(f"  Known REAL images -> avg P(FAKE)={mean_real_pfake:.4f}, avg P(REAL)={mean_real_preal:.4f}")
    print(f"  Separation gap (avg P(REAL|real) - avg P(REAL|fake)): {mean_real_preal - mean_fake_preal:+.4f}")
    print(f"  StdDev of P(REAL) across all 40 images: {std_preal_all:.4f}")
    print(f"  Mean max-class confidence:              {mean_conf:.2%}")
    print(f"  Predictions in 45%-55% band:            {near_fifty_count}/{len(y_pred)} ({near_fifty_count / len(y_pred):.1%})")
    print(f"  Predictions below 65% threshold:        {low_conf_count}/{len(y_pred)} ({low_conf_count / len(y_pred):.1%})")
    print(f"  Predicted class distribution:           FAKE={pred_fake_total}, REAL={pred_real_total}")

    one_class_bias = dominant_share >= 0.80
    print(
        f"  Single-class prediction bias detected?  {'YES' if one_class_bias else 'NO'} "
        f"(dominant predicted share = {dominant_share:.1%})"
    )

    prep_warnings = check_preprocessing_consistency(ROOT)

    # ------------------------------------------------------------------
    # Automated Root-Cause Summary & Next Correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIAGNOSTIC VERDICT & RECOMMENDED CORRECTION")
    print("=" * 72)

    auc = metrics["roc_auc"]
    if prep_warnings:
        print("  [ISSUE] Preprocessing inconsistencies detected:")
        for w_msg in prep_warnings:
            print(f"    - {w_msg}")

    if auc is not None and auc < 0.40:
        print(
            "  [ROOT CAUSE] INVERTED CLASS LABELS OR WEIGHTS: ROC-AUC is below 0.40, "
            "meaning real images score higher on class 0 and fake images score higher on class 1."
        )
        print("  [ACTION] Verify training folder class_to_idx and re-verify label alignment.")
    elif std_preal_all < 0.03 or weight_diff < 1e-3:
        print(
            "  [ROOT CAUSE] COLLAPSED / NEAR-CONSTANT OUTPUTS: Probability spread across images "
            f"is tiny (std={std_preal_all:.4f}), causing ~50/50 outputs regardless of image content."
        )
        print(
            "  [ACTION] Retrain with regularized two-stage transfer learning (scripts/train_model.py) "
            "using head warmup + differential LR + sufficient balanced samples."
        )
    elif one_class_bias:
        dominant_label = "FAKE" if pred_fake_total > pred_real_total else "REAL"
        print(
            f"  [ROOT CAUSE] SEVERE CLASS BIAS TOWARD {dominant_label}: {dominant_share:.1%} of test "
            "predictions collapsed to a single class."
        )
        print(
            "  [ACTION] Check train/valid class balance and retrain using balanced sampling and "
            "validation-loss checkpoint selection (`python scripts/train_model.py`)."
        )
    elif mean_conf < 0.65 or metrics["accuracy"] < 0.80:
        print(
            "  [ROOT CAUSE] UNDER-TRAINED / WEAKLY SEPARATED MODEL: "
            f"Accuracy is {metrics['accuracy']:.2%} with mean confidence {mean_conf:.2%} and "
            f"class separation gap {mean_real_preal - mean_fake_preal:+.4f}. "
            "The model lacks discriminative certainty on unseen faces, producing near-50/50 probabilities."
        )
        print(
            "  [ACTION] Do NOT lower the confidence threshold. Retrain EfficientNet-B0 on a larger "
            "balanced subset (or full 100k train / 20k valid) using the fixed two-stage pipeline:\n"
            "      python scripts/train_model.py --epochs 12 --batch-size 32\n"
            "    Then verify on unseen test images before replacing models/best_model.pth:\n"
            "      python scripts/evaluate_model.py --model-path models/retrained_best_model.pth --full"
        )
    else:
        print(
            f"  [RESULT] Checkpoint passed 40-image diagnostic (Accuracy={metrics['accuracy']:.2%}, "
            f"ROC-AUC={auc:.4f}, mean confidence={mean_conf:.2%})."
        )
        print(
            "  If your web API still gives ~50/50 predictions, inspect the running FastAPI server "
            "to ensure it loads this exact models/best_model.pth and uses backend/preprocessing.py."
        )

    if args.api_url:
        parity_samples = selected[:3] + selected[n_fake : n_fake + 3]
        compare_with_fastapi(args.api_url, model, transform, device, parity_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
