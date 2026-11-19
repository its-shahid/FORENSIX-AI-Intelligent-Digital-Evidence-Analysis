"""Hard-example analysis for Aegis Forensics EfficientNet-B0.

Identifies and characterizes FAKE images that models/best_model.pth misclassifies
as REAL (False Negatives = Actual FAKE predicted as REAL) across:
  - Dataset/train (safe hard-example mining for fine-tuning, zero test leakage)
  - Dataset/valid (validation false-negative audit)
(Dataset/test is intentionally NOT touched during hard-example mining.)

Outputs:
  - results/hard_false_negatives.json (missed FAKE paths, probabilities, & image cues)
  - results/hard_false_negatives_grid.png (visual contact sheet of top missed FAKEs)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASS_MAPPING = {"fake": 0, "real": 1}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexedPathDataset(Dataset):
    """Return preprocessed tensor plus original file path and label."""

    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        path_str, label = self.samples[idx]
        with Image.open(path_str) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        return self.transform(rgb), label, path_str


def compute_image_cues(path: Path) -> dict[str, float | int]:
    """Measure resolution, compression density, brightness, contrast, and high-freq edge variance."""
    size_bytes = path.stat().st_size
    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        w, h = rgb.size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # 3x3 discrete Laplacian energy (measures sharpness / high-frequency detail)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / max(1, h), 3),
        "bytes_per_pixel": round(size_bytes / max(1, w * h), 4),
        "brightness_mean": round(float(gray.mean()), 4),
        "contrast_std": round(float(gray.std()), 4),
        "laplacian_var": round(float(lap.var()), 6),
    }


def summarize_cues(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    keys = ("width", "height", "bytes_per_pixel", "brightness_mean", "contrast_std", "laplacian_var")
    summary: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in records if k in r]
        if vals:
            summary[f"mean_{k}"] = round(statistics.mean(vals), 5)
    return summary


def load_model(model_path: Path) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if ckpt.get("class_to_idx") != EXPECTED_CLASS_MAPPING:
        raise ValueError(f"Unexpected class_to_idx: {ckpt.get('class_to_idx')}")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def analyze_split(
    model: nn.Module,
    split_dir: Path,
    transform: transforms.Compose,
    batch_size: int = 32,
    max_cue_samples: int = 250,
) -> dict:
    folder = datasets.ImageFolder(split_dir)
    if folder.class_to_idx != EXPECTED_CLASS_MAPPING:
        raise ValueError(f"{split_dir} class_to_idx {folder.class_to_idx} != {EXPECTED_CLASS_MAPPING}")

    fake_samples = [(p, y) for p, y in folder.samples if y == 0]
    loader = DataLoader(
        IndexedPathDataset(fake_samples, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    missed_fakes: list[dict] = []
    caught_fakes: list[dict] = []

    with torch.inference_mode():
        for images, _labels, paths in loader:
            logits = model(images.to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            fake_probs = probs[:, 0].cpu().tolist()
            real_probs = probs[:, 1].cpu().tolist()

            for p_str, pf, pr in zip(paths, fake_probs, real_probs):
                entry = {
                    "path": p_str,
                    "p_fake": round(float(pf), 6),
                    "p_real": round(float(pr), 6),
                }
                if pr >= pf:
                    missed_fakes.append(entry)
                else:
                    caught_fakes.append(entry)

    # Sort worst false negatives first (highest P(REAL) on actual FAKE)
    missed_fakes.sort(key=lambda r: r["p_real"], reverse=True)

    missed_cues = [compute_image_cues(Path(r["path"])) for r in missed_fakes[:max_cue_samples]]
    caught_cues = [compute_image_cues(Path(r["path"])) for r in caught_fakes[:max_cue_samples]]

    total_fake = len(fake_samples)
    fn_count = len(missed_fakes)
    fake_recall = (total_fake - fn_count) / total_fake if total_fake else 0.0

    return {
        "split": split_dir.name,
        "total_fake_images": total_fake,
        "caught_fake_count": len(caught_fakes),
        "false_negative_count": fn_count,
        "fake_recall": round(fake_recall, 6),
        "missed_cue_summary": summarize_cues(missed_cues),
        "caught_cue_summary": summarize_cues(caught_cues),
        "false_negatives": missed_fakes,
    }


def save_contact_sheet(records: list[dict], out_path: Path, max_images: int = 16) -> None:
    subset = records[:max_images]
    if not subset:
        return
    cols = 4
    rows = (len(subset) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, rec in zip(axes_flat, subset):
        p = Path(rec["path"])
        with Image.open(p) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
        ax.imshow(rgb)
        ax.set_title(
            f"{p.name[:18]}\nP(FAKE)={rec['p_fake']:.1%} | P(REAL)={rec['p_real']:.1%}",
            fontsize=8.5,
        )

    plt.suptitle("Hard False Negatives (Actual FAKE Predicted as REAL)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify FAKE images misclassified as REAL.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only analyze Dataset/valid (faster diagnostic; omit to also mine hard training fakes).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    model = load_model(args.model_path)
    valid_dir = DATASET_PATH / "valid"
    train_dir = DATASET_PATH / "train"

    if not valid_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

    print(f"Running hard-example analysis using {args.model_path} on {DEVICE}...")
    valid_report = analyze_split(model, valid_dir, transform, batch_size=args.batch_size)
    print(
        f"[valid] FAKE total={valid_report['total_fake_images']} | "
        f"False Negatives (FAKE->REAL)={valid_report['false_negative_count']} | "
        f"Fake Recall={valid_report['fake_recall']:.2%}"
    )

    train_hard_paths: list[str] = []
    train_report: dict | None = None
    if not args.skip_train and train_dir.exists():
        train_report = analyze_split(model, train_dir, transform, batch_size=args.batch_size)
        train_hard_paths = [r["path"] for r in train_report["false_negatives"]]
        print(
            f"[train] FAKE total={train_report['total_fake_images']} | "
            f"Hard Fake False Negatives={train_report['false_negative_count']} | "
            f"Fake Recall={train_report['fake_recall']:.2%}"
        )

    grid_path = RESULTS_DIR / "hard_false_negatives_grid.png"
    save_contact_sheet(valid_report["false_negatives"], grid_path)

    payload = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.model_path),
        "class_mapping": EXPECTED_CLASS_MAPPING,
        "valid_summary": {k: v for k, v in valid_report.items() if k != "false_negatives"},
        "valid_false_negatives_top200": valid_report["false_negatives"][:200],
        "train_summary": (
            {k: v for k, v in train_report.items() if k != "false_negatives"}
            if train_report
            else None
        ),
        "train_hard_fake_paths": train_hard_paths,
    }
    out_json = RESULTS_DIR / "hard_false_negatives.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\nImage property comparison on Dataset/valid (Missed FAKE vs Caught FAKE):")
    print("  Missed FAKEs:", valid_report["missed_cue_summary"])
    print("  Caught FAKEs:", valid_report["caught_cue_summary"])
    print(f"\nSaved hard-example manifest to: {out_json}")
    if valid_report["false_negatives"]:
        print(f"Saved visual contact sheet to: {grid_path}")
    print(
        "\nNext step — fine-tune preserving models/best_model.pth with hard-example oversampling:\n"
        f"  python scripts/train_model.py --init-checkpoint {args.model_path} "
        f"--warmup-epochs 0 --epochs 8 --backbone-lr 2e-5 --head-lr 5e-5 "
        f"--hard-examples-json {out_json} --output models/improved_best_model.pth"
    )


if __name__ == "__main__":
    main()
