"""Comprehensive full-test evaluation, Precision-Recall curve, and model comparison script.

Evaluates models/best_model.pth (or any checkpoint via --model-path) on the complete
unseen Dataset/test split (or balanced subset via --sample-size) using exact FastAPI
inference preprocessing parity:
  ImageOps.exif_transpose -> RGB -> Resize((224, 224)) -> ToTensor -> ImageNet Normalize

Computes and saves:
  - Confusion Matrix (results/confusion_matrix.png)
  - Precision-Recall Curve for FAKE class (results/precision_recall_curve.png)
  - Accuracy, Fake Precision & Recall, Real Precision & Recall, Macro F1
  - False Negative Rate (FNR = fake missed as real) & False Positive Rate (FPR = real flagged fake)
  - ROC-AUC, PR-AUC (Average Precision for FAKE), Brier Score
  - Optional --compare-with <path> side-by-side comparison on the identical test split
    (results/model_comparison.json & results/comparison_confusion_matrices.png)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "Dataset"
MODEL_PATH = ROOT / "models" / "best_model.pth"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
DEFAULT_LOW_CONF_THRESHOLD = 0.65
DEFAULT_DECISION_THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["fake", "real"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def exif_rgb_loader(path: str) -> Image.Image:
    """Decode image with EXIF orientation applied and converted to 3-channel RGB."""
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


def build_eval_transform(use_center_crop: bool = False) -> transforms.Compose:
    steps: list[object] = []
    if use_center_crop:
        steps.append(transforms.Lambda(center_crop_square))
    steps.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def audit_dataset_splits(
    dataset_root: Path,
    check_hashes: bool = False,
    hash_sample_limit: int = 2000,
) -> dict:
    """Inspect class counts, imbalance ratios, and filename/content overlap across splits."""
    splits_info: dict[str, dict] = {}
    filenames_by_split: dict[str, set[str]] = {}
    hashes_by_split: dict[str, set[str]] = {}

    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            splits_info[split] = {"exists": False}
            continue

        fake_files = [
            p for p in (split_dir / "fake").rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ] if (split_dir / "fake").exists() else []
        real_files = [
            p for p in (split_dir / "real").rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ] if (split_dir / "real").exists() else []

        n_fake = len(fake_files)
        n_real = len(real_files)
        total = n_fake + n_real
        ratio = (
            round(max(n_fake, n_real) / min(n_fake, n_real), 4)
            if min(n_fake, n_real) > 0
            else None
        )
        splits_info[split] = {
            "exists": True,
            "fake": n_fake,
            "real": n_real,
            "total": total,
            "fake_share": round(n_fake / total, 4) if total else None,
            "real_share": round(n_real / total, 4) if total else None,
            "imbalance_ratio": ratio,
            "balanced": bool(ratio is not None and ratio <= 1.05),
        }

        all_files = fake_files + real_files
        filenames_by_split[split] = {f"{p.parent.name}/{p.name}" for p in all_files}

        if check_hashes and all_files:
            rng = random.Random(42)
            sampled = (
                all_files
                if len(all_files) <= hash_sample_limit
                else rng.sample(all_files, hash_sample_limit)
            )
            hashes_by_split[split] = {
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sampled
            }

    leakage: dict[str, dict] = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if a in filenames_by_split and b in filenames_by_split:
            pair_info: dict[str, int] = {
                "shared_relative_names": len(
                    filenames_by_split[a] & filenames_by_split[b]
                )
            }
            if check_hashes and a in hashes_by_split and b in hashes_by_split:
                pair_info["shared_sha256_in_sample"] = len(
                    hashes_by_split[a] & hashes_by_split[b]
                )
            leakage[f"{a}_vs_{b}"] = pair_info

    return {
        "splits": splits_info,
        "leakage_checks": leakage,
    }


def load_checkpoint_model(model_path: Path, expected_mapping: dict[str, int]) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError(
            f"Model '{model_path}' class_to_idx {checkpoint.get('class_to_idx')} and "
            f"dataset {expected_mapping} differ."
        )

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def _run_inference_on_loader(
    model: nn.Module,
    test_loader: DataLoader,
    total_samples: int,
    decision_threshold: float,
    low_conf_threshold: float,
) -> dict:
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_fake_probs: list[float] = []
    all_real_probs: list[float] = []
    all_confidences: list[float] = []
    start = time.perf_counter()

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            fake_probs = probs[:, 0].cpu().tolist()
            real_probs = probs[:, 1].cpu().tolist()

            for fp, rp in zip(fake_probs, real_probs):
                is_fake = (
                    fp > rp
                    if abs(decision_threshold - 0.5) < 1e-9
                    else fp >= decision_threshold
                )
                pred = 0 if is_fake else 1
                conf = fp if is_fake else rp
                all_predictions.append(pred)
                all_fake_probs.append(fp)
                all_real_probs.append(rp)
                all_confidences.append(conf)

            all_labels.extend(labels.tolist())
            if batch_number % 40 == 0:
                print(f"  Processed {len(all_labels)}/{total_samples} images")

    elapsed_sec = time.perf_counter() - start
    accuracy = float(accuracy_score(all_labels, all_predictions))

    roc_auc: float | None = None
    pr_auc_fake: float | None = None
    brier: float | None = None
    pr_curve_data: tuple[list[float], list[float]] | None = None

    if len(set(all_labels)) == 2:
        roc_auc = float(roc_auc_score(all_labels, all_real_probs))
        brier = float(brier_score_loss(all_labels, all_real_probs))
        fake_binary_labels = [1 if y == 0 else 0 for y in all_labels]
        pr_auc_fake = float(average_precision_score(fake_binary_labels, all_fake_probs))
        p_curve, r_curve, _ = precision_recall_curve(fake_binary_labels, all_fake_probs)
        pr_curve_data = (p_curve.tolist(), r_curve.tolist())

    prec, rec, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=[0, 1],
        zero_division=0,
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tp_fake, fn_fake = int(matrix[0, 0]), int(matrix[0, 1])
    fp_real, tn_real = int(matrix[1, 0]), int(matrix[1, 1])

    fnr = fn_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) else 0.0
    fpr = fp_real / (fp_real + tn_real) if (fp_real + tn_real) else 0.0

    fake_true_pfake = [pf for y, pf in zip(all_labels, all_fake_probs) if y == 0]
    real_true_pfake = [pf for y, pf in zip(all_labels, all_fake_probs) if y == 1]

    low_conf_count = sum(1 for c in all_confidences if c < low_conf_threshold)
    near_fifty = sum(1 for c in all_confidences if c < 0.55) / len(all_confidences)

    return {
        "all_labels": all_labels,
        "all_predictions": all_predictions,
        "pr_curve_data": pr_curve_data,
        "matrix": matrix,
        "summary": {
            "evaluated_samples": len(all_labels),
            "decision_threshold_fake": decision_threshold,
            "accuracy": round(accuracy, 6),
            "precision_macro": round(float(macro_prec), 6),
            "recall_macro": round(float(macro_rec), 6),
            "f1_macro": round(float(macro_f1), 6),
            "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
            "pr_auc_fake": round(pr_auc_fake, 6) if pr_auc_fake is not None else None,
            "brier_score": round(brier, 6) if brier is not None else None,
            "false_negative_rate_fnr": round(fnr, 6),
            "false_positive_rate_fpr": round(fpr, 6),
            "fake_missed_as_real_count": fn_fake,
            "real_flagged_as_fake_count": fp_real,
            "mean_p_fake_on_true_fake": round(sum(fake_true_pfake) / len(fake_true_pfake), 6) if fake_true_pfake else None,
            "mean_p_fake_on_true_real": round(sum(real_true_pfake) / len(real_true_pfake), 6) if real_true_pfake else None,
            "low_confidence_warning_threshold": low_conf_threshold,
            "low_confidence_count": low_conf_count,
            "low_confidence_rate": round(low_conf_count / len(all_confidences), 6),
            "near_fifty_rate": round(near_fifty, 6),
            "per_class": {
                "fake": {
                    "class_id": 0,
                    "precision": round(float(prec[0]), 6),
                    "recall": round(float(rec[0]), 6),
                    "f1": round(float(f1[0]), 6),
                    "support": int(support[0]),
                },
                "real": {
                    "class_id": 1,
                    "precision": round(float(prec[1]), 6),
                    "recall": round(float(rec[1]), 6),
                    "f1": round(float(f1[1]), 6),
                    "support": int(support[1]),
                },
            },
            "confusion_matrix": matrix.tolist(),
            "avg_confidence": round(sum(all_confidences) / len(all_confidences), 6),
            "avg_inference_ms_per_image": round((elapsed_sec * 1000.0) / len(all_labels), 2),
        },
    }


def _save_confusion_matrix_plot(matrix, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred FAKE (0)", "Pred REAL (1)"],
        yticklabels=["Actual FAKE (0)", "Actual REAL (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_pr_curve_plot(
    curves: list[tuple[str, list[float], list[float], float | None, float, float]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.5, 5.5))
    for label_name, prec_pts, rec_pts, ap, op_rec, op_prec in curves:
        ap_str = f"AP={ap:.4f}" if ap is not None else "AP=n/a"
        plt.plot(rec_pts, prec_pts, linewidth=2.0, label=f"{label_name} ({ap_str})")
        plt.scatter([op_rec], [op_prec], s=60, zorder=5)
    plt.xlabel("FAKE Recall (Sensitivity)")
    plt.ylabel("FAKE Precision")
    plt.title("Precision-Recall Curve (Positive Class = FAKE)")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _print_summary(model_name: str, res: dict) -> None:
    s = res["summary"]
    print(f"\nRESULTS FOR {model_name} (decision P(FAKE) >= {s['decision_threshold_fake']:.2f})")
    print("=" * 60)
    print(f"  Evaluated samples: {s['evaluated_samples']}")
    print(f"  Accuracy:          {s['accuracy']:.2%}")
    print(f"  Macro F1-Score:    {s['f1_macro']:.4f}")
    print(f"  FAKE Precision:    {s['per_class']['fake']['precision']:.2%}")
    print(f"  FAKE Recall:       {s['per_class']['fake']['recall']:.2%}  (FNR / Missed Fake: {s['false_negative_rate_fnr']:.2%}, count={s['fake_missed_as_real_count']})")
    print(f"  REAL Precision:    {s['per_class']['real']['precision']:.2%}")
    print(f"  REAL Recall:       {s['per_class']['real']['recall']:.2%}  (FPR / False Alarm: {s['false_positive_rate_fpr']:.2%}, count={s['real_flagged_as_fake_count']})")
    if s["roc_auc"] is not None:
        print(f"  ROC-AUC:           {s['roc_auc']:.4f}")
    if s["pr_auc_fake"] is not None:
        print(f"  PR-AUC (FAKE):     {s['pr_auc_fake']:.4f}")
    if s["brier_score"] is not None:
        print(f"  Brier Score:       {s['brier_score']:.4f}")
    print(f"  Low-Conf (<{s['low_confidence_warning_threshold']:.2f}):  {s['low_confidence_count']}/{s['evaluated_samples']} ({s['low_confidence_rate']:.2%})")
    print("\n  Classification Report:")
    print(
        classification_report(
            res["all_labels"],
            res["all_predictions"],
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print("  Confusion Matrix (rows=Actual [fake, real], cols=Predicted [fake, real]):")
    print(res["matrix"])


def evaluate(
    sample_size: int | None = None,
    seed: int = 42,
    model_path: Path = MODEL_PATH,
    compare_with: Path | None = None,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD,
    check_hashes: bool = False,
    use_center_crop: bool = False,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_dir = DATASET_PATH / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset folder not found: {test_dir}")

    dataset_audit = audit_dataset_splits(DATASET_PATH, check_hashes=check_hashes)
    transform = build_eval_transform(use_center_crop=use_center_crop)

    full_test_data = datasets.ImageFolder(
        test_dir,
        transform=transform,
        loader=exif_rgb_loader,
    )
    if full_test_data.class_to_idx != {"fake": 0, "real": 1}:
        raise ValueError(
            f"Unexpected class mapping: {full_test_data.class_to_idx}. Expected {{'fake': 0, 'real': 1}}."
        )

    fake_indices = [i for i, (_, label) in enumerate(full_test_data.samples) if label == 0]
    real_indices = [i for i, (_, label) in enumerate(full_test_data.samples) if label == 1]

    if sample_size is None:
        selected_indices = list(range(len(full_test_data)))
    else:
        per_class = sample_size // 2
        if len(fake_indices) < per_class or len(real_indices) < per_class:
            per_class = min(len(fake_indices), len(real_indices))
        if per_class == 0:
            raise ValueError(
                f"Not enough images in classes (fake={len(fake_indices)}, real={len(real_indices)})."
            )
        rng = random.Random(seed)
        selected_indices = rng.sample(fake_indices, per_class) + rng.sample(real_indices, per_class)
        rng.shuffle(selected_indices)

    if not selected_indices:
        raise ValueError("No test images were found.")

    test_data = Subset(full_test_data, selected_indices)
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nDATASET SPLIT AUDIT")
    print("=" * 60)
    for split_name, info in dataset_audit["splits"].items():
        if info.get("exists"):
            print(
                f"{split_name:>5}: total={info['total']} | fake={info['fake']} | "
                f"real={info['real']} | imbalance_ratio={info['imbalance_ratio']}"
            )
    for pair_name, pair_info in dataset_audit["leakage_checks"].items():
        print(f"Leakage ({pair_name}): {pair_info}")

    print(f"\nEvaluating primary checkpoint: {model_path} on {len(test_data)} unseen test images...")
    primary_model = load_checkpoint_model(model_path, full_test_data.class_to_idx)
    primary_res = _run_inference_on_loader(
        primary_model,
        test_loader,
        len(test_data),
        decision_threshold=decision_threshold,
        low_conf_threshold=low_conf_threshold,
    )
    _print_summary(model_path.name, primary_res)

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    _save_confusion_matrix_plot(
        primary_res["matrix"],
        f"Confusion Matrix ({model_path.name} · Acc={primary_res['summary']['accuracy']:.2%})",
        cm_path,
    )

    pr_curves_to_plot = []
    if primary_res["pr_curve_data"] is not None:
        p_pts, r_pts = primary_res["pr_curve_data"]
        pr_curves_to_plot.append(
            (
                model_path.name,
                p_pts,
                r_pts,
                primary_res["summary"]["pr_auc_fake"],
                primary_res["summary"]["per_class"]["fake"]["recall"],
                primary_res["summary"]["per_class"]["fake"]["precision"],
            )
        )

    s = primary_res["summary"]
    metrics_payload = {
        "architecture": "EfficientNet-B0",
        "checkpoint": str(model_path.name),
        "dataset": "Dataset/test",
        "class_mapping": {"fake": 0, "real": 1},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_test_available": len(full_test_data),
        "dataset_audit": dataset_audit,
        **s,
        "device": str(DEVICE),
    }

    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    if compare_with is not None:
        print(f"\nEvaluating comparison checkpoint: {compare_with} on identical {len(test_data)} test images...")
        comp_model = load_checkpoint_model(compare_with, full_test_data.class_to_idx)
        comp_res = _run_inference_on_loader(
            comp_model,
            test_loader,
            len(test_data),
            decision_threshold=decision_threshold,
            low_conf_threshold=low_conf_threshold,
        )
        _print_summary(compare_with.name, comp_res)

        if comp_res["pr_curve_data"] is not None:
            cp_pts, cr_pts = comp_res["pr_curve_data"]
            pr_curves_to_plot.append(
                (
                    compare_with.name,
                    cp_pts,
                    cr_pts,
                    comp_res["summary"]["pr_auc_fake"],
                    comp_res["summary"]["per_class"]["fake"]["recall"],
                    comp_res["summary"]["per_class"]["fake"]["precision"],
                )
            )

        cs = comp_res["summary"]
        print("\nSIDE-BY-SIDE TEST COMPARISON")
        print("=" * 72)
        rows = [
            ("Accuracy", s["accuracy"], cs["accuracy"], False),
            ("FAKE Recall", s["per_class"]["fake"]["recall"], cs["per_class"]["fake"]["recall"], False),
            ("FAKE Precision", s["per_class"]["fake"]["precision"], cs["per_class"]["fake"]["precision"], False),
            ("False Negative Count (FN)", s["fake_missed_as_real_count"], cs["fake_missed_as_real_count"], True),
            ("False Negative Rate (FNR)", s["false_negative_rate_fnr"], cs["false_negative_rate_fnr"], False),
            ("REAL Recall", s["per_class"]["real"]["recall"], cs["per_class"]["real"]["recall"], False),
            ("REAL Precision", s["per_class"]["real"]["precision"], cs["per_class"]["real"]["precision"], False),
            ("False Positive Count (FP)", s["real_flagged_as_fake_count"], cs["real_flagged_as_fake_count"], True),
            ("False Positive Rate (FPR)", s["false_positive_rate_fpr"], cs["false_positive_rate_fpr"], False),
            ("Macro F1-Score", s["f1_macro"], cs["f1_macro"], False),
            ("ROC-AUC", s["roc_auc"], cs["roc_auc"], False),
            ("PR-AUC (FAKE)", s["pr_auc_fake"], cs["pr_auc_fake"], False),
        ]
        print(f"{'Metric':<28} {model_path.name:<16} {compare_with.name:<16} {'Delta':>10}")
        print("-" * 72)
        for name, v1, v2, is_int in rows:
            if v1 is None or v2 is None:
                print(f"{name:<28} {'n/a':<16} {'n/a':<16} {'n/a':>10}")
            elif is_int:
                print(f"{name:<28} {int(v1):<16d} {int(v2):<16d} {int(v2) - int(v1):>+10d}")
            else:
                print(f"{name:<28} {v1:<16.4f} {v2:<16.4f} {v2 - v1:>+10.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, m_res, m_name in (
            (axes[0], primary_res, model_path.name),
            (axes[1], comp_res, compare_with.name),
        ):
            sns.heatmap(
                m_res["matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred FAKE", "Pred REAL"],
                yticklabels=["Actual FAKE", "Actual REAL"],
                ax=ax,
            )
            ax.set_title(f"{m_name} (Acc={m_res['summary']['accuracy']:.2%})")
        plt.tight_layout()
        comp_cm_path = RESULTS_DIR / "comparison_confusion_matrices.png"
        plt.savefig(comp_cm_path, dpi=150)
        plt.close()

        comparison_payload = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "evaluated_samples": len(test_data),
            "decision_threshold_fake": decision_threshold,
            "baseline_model": {"checkpoint": model_path.name, **s},
            "candidate_model": {"checkpoint": compare_with.name, **cs},
        }
        comp_json_path = RESULTS_DIR / "model_comparison.json"
        with comp_json_path.open("w", encoding="utf-8") as fh:
            json.dump(comparison_payload, fh, indent=2)
        print(f"\nComparison confusion matrices saved to: {comp_cm_path}")
        print(f"Comparison metrics JSON saved to:      {comp_json_path}")

    if pr_curves_to_plot:
        pr_path = RESULTS_DIR / "precision_recall_curve.png"
        _save_pr_curve_plot(pr_curves_to_plot, pr_path)
        print(f"Precision-Recall curve saved to:       {pr_path}")

    print(f"Confusion matrix saved to:             {cm_path}")
    print(f"Structured metrics saved to:           {metrics_path}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet-B0 checkpoint on unseen Dataset/test."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional balanced sample size across fake/real (default: None = FULL Dataset/test).",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Decision threshold for predicting FAKE: P(FAKE) >= t (default: 0.50).",
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=DEFAULT_LOW_CONF_THRESHOLD,
        help="Low-confidence warning cutoff to audit (default: 0.65).",
    )
    parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Also hash sampled files across train/valid/test to detect duplicate image leakage.",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Optionally square-crop rectangular images before 224x224 resize.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to primary checkpoint file (default: models/best_model.pth).",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Optional second checkpoint (e.g., models/improved_best_model.pth) for side-by-side comparison.",
    )
    args = parser.parse_args()

    evaluate(
        sample_size=args.sample_size,
        model_path=args.model_path,
        compare_with=args.compare_with,
        decision_threshold=args.decision_threshold,
        low_conf_threshold=args.low_conf_threshold,
        check_hashes=args.check_hashes,
        use_center_crop=args.center_crop,
    )


if __name__ == "__main__":
    main()
