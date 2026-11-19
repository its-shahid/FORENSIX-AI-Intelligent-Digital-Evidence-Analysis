"""Improved EfficientNet-B0 training pipeline for Aegis Forensics.

Preserves existing architecture (torchvision EfficientNet-B0 with nn.Linear(1280, 2)),
input size (224x224 RGB), ImageNet normalization, and class mapping (fake=0, real=1),
while addressing false-negative AI fake misclassifications through:
  1. Exact preprocessing parity with FastAPI inference (EXIF transpose + Resize(224, 224)
     + ToTensor + ImageNet Normalize; optional --center-crop only if desired).
  2. Support for mixing modern synthetic face generators (--extra-fake-dirs) and diverse
     real photos (--extra-real-dirs) alongside Dataset/train with deterministic split
     isolation and optional SHA-256 validation deduplication (--deduplicate-valid).
  3. Artifact-preserving augmentations (horizontal flip, subtle affine, mild color jitter,
     light JPEG recompression, light blur, and optional RandomErasing) so the model learns
     generalizable real-vs-AI texture/frequency cues without erasing synthetic artifacts.
  4. Configurable Cross-Entropy or Focal Loss (--loss ce|focal) and optional FAKE class
     weighting (--fake-class-weight) to reduce fake-missed-as-real false negatives.
  5. Two-stage transfer learning (head warmup + differential LR cosine fine-tuning) with
     validation tracking of Fake Recall, FNR, Real Recall, FPR, Macro F1, ROC-AUC, and PR-AUC.
  6. Safe output path defaulting to models/improved_best_model.pth (never overwrites
     models/best_model.pth).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from PIL import Image, ImageOps
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASS_MAPPING = {"fake": 0, "real": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomJPEGCompression:
    """Simulate real-world web/social JPEG recompression during training."""

    def __init__(self, p: float = 0.25, quality_range: tuple[int, int] = (70, 96)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0 or random.random() >= self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def exif_rgb_loader(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 3 if height > width else (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PathLabelDataset(Dataset):
    """Lightweight image dataset supporting combined ImageFolder + extra directories."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = exif_rgb_loader(path)
        return self.transform(image), label


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal_factor = (1.0 - pt) ** self.gamma
        return (focal_factor * ce).mean()


def build_transforms(
    use_center_crop: bool = False,
    jpeg_prob: float = 0.25,
    blur_prob: float = 0.10,
    random_erasing_prob: float = 0.15,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[object] = []
    valid_steps: list[object] = []

    if use_center_crop:
        train_steps.append(transforms.Lambda(center_crop_square))
        valid_steps.append(transforms.Lambda(center_crop_square))

    train_steps.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.95, 1.05))],
                p=0.35,
            ),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
            RandomJPEGCompression(p=jpeg_prob, quality_range=(70, 96)),
        ]
    )
    if blur_prob > 0.0:
        train_steps.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=blur_prob,
            )
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if random_erasing_prob > 0.0:
        train_steps.append(
            transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.12), ratio=(0.3, 3.3))
        )

    valid_steps.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), transforms.Compose(valid_steps)


def collect_images_from_dirs(dirs: list[Path]) -> list[Path]:
    collected: list[Path] = []
    for folder in dirs:
        if not folder.exists():
            raise FileNotFoundError(f"Extra directory not found: {folder}")
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                collected.append(p)
    return collected


def balance_samples(
    samples: list[tuple[Path, int]],
    total_limit: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    fake_samples = [s for s in samples if s[1] == 0]
    real_samples = [s for s in samples if s[1] == 1]

    min_count = min(len(fake_samples), len(real_samples))
    if min_count == 0:
        raise ValueError(
            f"Cannot balance classes: fake={len(fake_samples)}, real={len(real_samples)}."
        )

    if total_limit is None:
        per_class = min_count
    else:
        requested_per_class = total_limit // 2
        if requested_per_class > min_count:
            raise ValueError(
                f"Requested {requested_per_class} per class, but smallest class has only {min_count}."
            )
        per_class = requested_per_class

    chosen = rng.sample(fake_samples, per_class) + rng.sample(real_samples, per_class)
    rng.shuffle(chosen)
    return chosen


def deduplicate_against_train(
    train_samples: list[tuple[Path, int]],
    valid_samples: list[tuple[Path, int]],
) -> tuple[list[tuple[Path, int]], int]:
    """Remove validation images whose SHA-256 hash matches any training image."""
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in train_samples}
    filtered_valid: list[tuple[Path, int]] = []
    removed = 0
    for p, lbl in valid_samples:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in train_hashes:
            removed += 1
        else:
            filtered_valid.append((p, lbl))
    return filtered_valid, removed


def load_data(
    batch_size: int,
    train_limit: int | None,
    valid_limit: int | None,
    seed: int,
    num_workers: int,
    use_center_crop: bool,
    jpeg_prob: float,
    blur_prob: float,
    random_erasing_prob: float,
    extra_fake_dirs: list[Path],
    extra_real_dirs: list[Path],
    extra_val_ratio: float,
    deduplicate_valid: bool,
    hard_examples_json: Path | None = None,
    hard_example_boost: float = 3.0,
):
    train_path = DATASET_PATH / "train"
    valid_path = DATASET_PATH / "valid"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Could not find train or valid folders in {DATASET_PATH}")

    train_folder = datasets.ImageFolder(train_path)
    valid_folder = datasets.ImageFolder(valid_path)

    if train_folder.class_to_idx != EXPECTED_CLASS_MAPPING:
        raise ValueError(
            f"Expected class mapping {EXPECTED_CLASS_MAPPING}, got {train_folder.class_to_idx}"
        )
    if train_folder.class_to_idx != valid_folder.class_to_idx:
        raise ValueError("Training and validation class mappings differ.")

    train_samples: list[tuple[Path, int]] = [(Path(p), lbl) for p, lbl in train_folder.samples]
    valid_samples: list[tuple[Path, int]] = [(Path(p), lbl) for p, lbl in valid_folder.samples]

    rng = random.Random(seed)
    if extra_fake_dirs:
        extra_fakes = collect_images_from_dirs(extra_fake_dirs)
        rng.shuffle(extra_fakes)
        n_val = int(len(extra_fakes) * extra_val_ratio)
        valid_samples.extend((p, 0) for p in extra_fakes[:n_val])
        train_samples.extend((p, 0) for p in extra_fakes[n_val:])
        print(
            f"Added extra FAKE images: {len(extra_fakes) - n_val} train, {n_val} valid "
            f"(from {len(extra_fake_dirs)} dir(s))"
        )

    if extra_real_dirs:
        extra_reals = collect_images_from_dirs(extra_real_dirs)
        rng.shuffle(extra_reals)
        n_val = int(len(extra_reals) * extra_val_ratio)
        valid_samples.extend((p, 1) for p in extra_reals[:n_val])
        train_samples.extend((p, 1) for p in extra_reals[n_val:])
        print(
            f"Added extra REAL images: {len(extra_reals) - n_val} train, {n_val} valid "
            f"(from {len(extra_real_dirs)} dir(s))"
        )

    if deduplicate_valid:
        valid_samples, removed_dups = deduplicate_against_train(train_samples, valid_samples)
        print(f"SHA-256 deduplication removed {removed_dups} duplicate valid image(s) present in train.")

    balanced_train = balance_samples(train_samples, train_limit, seed=seed)
    balanced_valid = balance_samples(valid_samples, valid_limit, seed=seed + 1)

    train_tf, valid_tf = build_transforms(
        use_center_crop=use_center_crop,
        jpeg_prob=jpeg_prob,
        blur_prob=blur_prob,
        random_erasing_prob=random_erasing_prob,
    )

    train_dataset = PathLabelDataset(balanced_train, transform=train_tf)
    valid_dataset = PathLabelDataset(balanced_valid, transform=valid_tf)

    train_sampler = None
    if hard_examples_json is not None:
        if not hard_examples_json.exists():
            raise FileNotFoundError(f"Hard-examples JSON not found: {hard_examples_json}")
        hard_payload = json.loads(hard_examples_json.read_text(encoding="utf-8"))
        hard_paths = {
            str(Path(p).resolve())
            for p in hard_payload.get("train_hard_fake_paths", [])
        }
        # Preserve overall 50/50 FAKE/REAL expected batch weight while oversampling hard FAKEs
        hard_fake_count = sum(
            1 for p, lbl in balanced_train if lbl == 0 and str(p.resolve()) in hard_paths
        )
        easy_fake_count = sum(
            1 for p, lbl in balanced_train if lbl == 0 and str(p.resolve()) not in hard_paths
        )
        total_fake_count = hard_fake_count + easy_fake_count
        if hard_fake_count > 0 and total_fake_count > 0:
            raw_fake_sum = hard_fake_count * hard_example_boost + easy_fake_count * 1.0
            norm_factor = total_fake_count / raw_fake_sum
            weights = [
                (hard_example_boost * norm_factor if str(p.resolve()) in hard_paths else norm_factor)
                if lbl == 0
                else 1.0
                for p, lbl in balanced_train
            ]
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(balanced_train),
                replacement=True,
            )
            print(
                f"Hard-example sampler active: boosting {hard_fake_count} hard FAKE training image(s) "
                f"by {hard_example_boost:.2f}x while maintaining 50/50 FAKE/REAL batch balance."
            )
        else:
            print("Hard-examples JSON had 0 matching training paths in current subset; using uniform shuffle.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Class mapping:", train_folder.class_to_idx)
    print(f"Balanced training images:   {len(balanced_train)} ({len(balanced_train) // 2}/class)")
    print(f"Balanced validation images: {len(balanced_valid)} ({len(balanced_valid) // 2}/class)")
    return train_loader, valid_loader, train_folder.class_to_idx


def build_model(dropout_p: float = 0.3, init_checkpoint: Path | None = None) -> nn.Module:
    if init_checkpoint is not None:
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location=DEVICE, weights_only=True)
        if ckpt.get("class_to_idx") != EXPECTED_CLASS_MAPPING:
            raise ValueError(
                f"Init checkpoint class_to_idx {ckpt.get('class_to_idx')} != {EXPECTED_CLASS_MAPPING}"
            )
        model = models.efficientnet_b0(weights=None)
        input_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout_p, inplace=True)
        model.classifier[1] = nn.Linear(input_features, 2)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Initialized EfficientNet-B0 weights from existing checkpoint: {init_checkpoint}")
        return model.to(DEVICE)

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    input_features = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=dropout_p, inplace=True)
    model.classifier[1] = nn.Linear(input_features, 2)
    return model.to(DEVICE)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_fake_probs: list[float] = []
    all_real_probs: list[float] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_fake_probs.extend(probs[:, 0].cpu().tolist())
            all_real_probs.extend(probs[:, 1].cpu().tolist())

    accuracy = sum(1 for y, p in zip(all_labels, all_preds) if y == p) / total if total else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )
    macro_f1 = float((f1[0] + f1[1]) / 2.0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tp_fake, fn_fake = int(cm[0, 0]), int(cm[0, 1])
    fp_real, tn_real = int(cm[1, 0]), int(cm[1, 1])
    fnr = fn_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) else 0.0
    fpr = fp_real / (fp_real + tn_real) if (fp_real + tn_real) else 0.0

    roc_auc: float | None = None
    pr_auc_fake: float | None = None
    if len(set(all_labels)) == 2:
        roc_auc = float(roc_auc_score(all_labels, all_real_probs))
        fake_binary_labels = [1 if y == 0 else 0 for y in all_labels]
        pr_auc_fake = float(average_precision_score(fake_binary_labels, all_fake_probs))

    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": accuracy,
        "fake_precision": float(prec[0]),
        "fake_recall": float(rec[0]),
        "fake_f1": float(f1[0]),
        "fnr": fnr,
        "real_precision": float(prec[1]),
        "real_recall": float(rec[1]),
        "real_f1": float(f1[1]),
        "fpr": fpr,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc_fake": pr_auc_fake,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved EfficientNet-B0 deepfake training pipeline.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Head-only warmup epochs before unfreezing backbone.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce", help="Loss function (ce or focal).")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma parameter when --loss focal.")
    parser.add_argument(
        "--fake-class-weight",
        type=float,
        default=1.0,
        help="Optional loss weight multiplier for FAKE class (0). Keep 1.0 unless emphasizing fake recall.",
    )
    parser.add_argument(
        "--select-by",
        choices=["macro_f1", "val_loss", "roc_auc"],
        default="macro_f1",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Total balanced training samples (None = full balanced train split).")
    parser.add_argument("--valid-limit", type=int, default=None, help="Total balanced validation samples (None = full balanced valid split).")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience after warmup.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--center-crop", action="store_true", help="Optionally square-crop before 224x224 resize.")
    parser.add_argument("--jpeg-prob", type=float, default=0.25, help="Probability of JPEG compression augmentation.")
    parser.add_argument("--blur-prob", type=float, default=0.10, help="Probability of light Gaussian blur augmentation.")
    parser.add_argument("--random-erasing-prob", type=float, default=0.15, help="Probability of RandomErasing cutout.")
    parser.add_argument("--extra-fake-dirs", type=Path, nargs="*", default=[], help="Extra directories of modern AI-generated fake images.")
    parser.add_argument("--extra-real-dirs", type=Path, nargs="*", default=[], help="Extra directories of diverse real images.")
    parser.add_argument("--extra-val-ratio", type=float, default=0.15, help="Validation split fraction for extra directories.")
    parser.add_argument("--deduplicate-valid", action="store_true", help="Drop exact SHA-256 duplicates of training images from validation.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional existing checkpoint (e.g. models/best_model.pth) to fine-tune from without overwriting it.",
    )
    parser.add_argument(
        "--hard-examples-json",
        type=Path,
        default=None,
        help="Optional JSON generated by scripts/hard_example_analysis.py to boost hard FAKE training samples.",
    )
    parser.add_argument(
        "--hard-example-boost",
        type=float,
        default=3.0,
        help="Sampling weight multiplier for mined hard FAKE training images (default: 3.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "improved_best_model.pth",
        help="Checkpoint destination (defaults to models/improved_best_model.pth to preserve models/best_model.pth).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print(f"Preprocessing: Resize({IMAGE_SIZE}, {IMAGE_SIZE}), center_crop={args.center_crop}, EXIF=True")
    print(f"Loss: {args.loss} | fake_class_weight={args.fake_class_weight} | select_by={args.select_by}")

    train_loader, valid_loader, class_to_idx = load_data(
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        valid_limit=args.valid_limit,
        seed=args.seed,
        num_workers=args.num_workers,
        use_center_crop=args.center_crop,
        jpeg_prob=args.jpeg_prob,
        blur_prob=args.blur_prob,
        random_erasing_prob=args.random_erasing_prob,
        extra_fake_dirs=args.extra_fake_dirs,
        extra_real_dirs=args.extra_real_dirs,
        extra_val_ratio=args.extra_val_ratio,
        deduplicate_valid=args.deduplicate_valid,
        hard_examples_json=args.hard_examples_json,
        hard_example_boost=args.hard_example_boost,
    )

    model = build_model(dropout_p=args.dropout, init_checkpoint=args.init_checkpoint)

    class_weights = None
    if abs(args.fake_class_weight - 1.0) > 1e-6:
        class_weights = torch.tensor([args.fake_class_weight, 1.0], dtype=torch.float32, device=DEVICE)

    if args.loss == "focal":
        train_criterion: nn.Module = FocalCrossEntropyLoss(
            gamma=args.focal_gamma,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        train_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    eval_criterion = nn.CrossEntropyLoss()

    best_score = float("inf") if args.select_by == "val_loss" else float("-inf")
    best_metrics: dict[str, float | None] | None = None
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        is_warmup = epoch < args.warmup_epochs
        for param in model.features.parameters():
            param.requires_grad = not is_warmup
        for param in model.classifier.parameters():
            param.requires_grad = True

        if epoch == args.warmup_epochs:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": args.backbone_lr},
                    {"params": model.classifier.parameters(), "lr": args.head_lr * 0.5},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
            )
            print("\n[Stage 2: Differential Fine-Tuning (features + classifier)]")
        elif args.warmup_epochs > 0 and epoch == 0:
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=args.head_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = None
            print(f"\n[Stage 1: Head Warmup for {args.warmup_epochs} epoch(s)]")

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = train_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        val = evaluate(model, valid_loader, eval_criterion)
        auc_str = f"{val['roc_auc']:.4f}" if val["roc_auc"] is not None else "n/a"
        pr_str = f"{val['pr_auc_fake']:.4f}" if val["pr_auc_fake"] is not None else "n/a"
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({'warmup' if is_warmup else 'finetune'}) | "
            f"train_loss={total_loss / total:.4f} train_acc={correct / total:.2%} | "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.2%} macro_f1={val['macro_f1']:.4f} | "
            f"fake_rec={val['fake_recall']:.2%} (FNR={val['fnr']:.2%}) "
            f"real_rec={val['real_recall']:.2%} (FPR={val['fpr']:.2%}) | "
            f"roc_auc={auc_str} pr_auc_fake={pr_str}"
        )

        if args.select_by == "val_loss":
            current_score = float(val["loss"])
            improved = current_score < best_score
        elif args.select_by == "roc_auc":
            current_score = float(val["roc_auc"] if val["roc_auc"] is not None else val["accuracy"])
            improved = current_score > best_score
        else:
            current_score = float(val["macro_f1"])
            improved = current_score > best_score

        if improved:
            best_score = current_score
            best_metrics = val
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "class_to_idx": class_to_idx,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch + 1,
                    "val_loss": round(float(val["loss"]), 6),
                    "val_accuracy": round(float(val["accuracy"]), 6),
                    "val_macro_f1": round(float(val["macro_f1"]), 6),
                    "val_fake_recall": round(float(val["fake_recall"]), 6),
                    "val_real_recall": round(float(val["real_recall"]), 6),
                    "val_fnr": round(float(val["fnr"]), 6),
                    "val_fpr": round(float(val["fpr"]), 6),
                    "val_roc_auc": round(float(val["roc_auc"]), 6) if val["roc_auc"] is not None else None,
                    "val_pr_auc_fake": round(float(val["pr_auc_fake"]), 6) if val["pr_auc_fake"] is not None else None,
                },
                args.output,
            )
            print(f"  -> Saved improved checkpoint ({args.select_by}={best_score:.4f}) to {args.output}")
        else:
            no_improve_epochs += 1
            if not is_warmup and no_improve_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without {args.select_by} improvement.")
                break

    if best_metrics is not None:
        print(
            f"\nTraining complete. Best {args.select_by}={best_score:.4f} | "
            f"val_acc={best_metrics['accuracy']:.2%}, fake_rec={best_metrics['fake_recall']:.2%}, "
            f"real_rec={best_metrics['real_recall']:.2%}, macro_f1={best_metrics['macro_f1']:.4f}"
        )
    print(
        "\nNext steps (without overwriting models/best_model.pth):\n"
        f"  1. Calibrate decision threshold on Dataset/valid:\n"
        f"     python scripts/calibrate_threshold.py --model-path {args.output}\n"
        f"  2. Compare original vs improved checkpoint on full unseen Dataset/test:\n"
        f"     python scripts/evaluate_model.py --model-path models/best_model.pth --compare-with {args.output}"
    )


if __name__ == "__main__":
    main()
