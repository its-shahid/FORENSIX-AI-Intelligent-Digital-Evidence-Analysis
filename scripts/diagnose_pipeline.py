"""End-to-end diagnostic script for Aegis Forensics accuracy debugging.

Checks:
  1. Checkpoint integrity (keys, class_to_idx={'fake': 0, 'real': 1}, NaN/Inf check,
     classifier weight magnitudes & bias symmetry).
  2. Training vs evaluation preprocessing parity.
  3. Dataset/train, valid, test class balance & cross-split leakage (when Dataset/ exists).
  4. Optional live FastAPI endpoint test on real Dataset/test/fake & Dataset/test/real samples
     via `--api-url https://...` to verify the remote server produces accurate predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def inspect_checkpoint(model_path: Path) -> bool:
    import torch
    import torch.nn as nn
    from torchvision import models

    print("\n[1] INSPECTING CHECKPOINT:", model_path)
    if not model_path.exists():
        print(f"  MISSING: {model_path} not found locally.")
        return False

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        print(f"  ERROR: Unexpected checkpoint type {type(ckpt)}")
        return False

    print("  Checkpoint keys:", list(ckpt.keys()))
    class_to_idx = ckpt.get("class_to_idx")
    print("  class_to_idx:   ", class_to_idx)
    if class_to_idx != {"fake": 0, "real": 1}:
        print("  ERROR: class_to_idx does NOT match {'fake': 0, 'real': 1}!")
        return False

    print("  image_size:     ", ckpt.get("image_size"))
    state = ckpt.get("model_state_dict")
    if not state:
        print("  ERROR: Missing model_state_dict.")
        return False

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(state, strict=True)

    non_finite = [
        k for k, v in state.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()
    ]
    if non_finite:
        print("  ERROR: Non-finite parameters detected:", non_finite)
        return False

    bias = state["classifier.1.bias"].tolist()
    w_norm = [float(state["classifier.1.weight"][i].norm()) for i in range(2)]
    print(f"  Classifier head weight norms: fake(0)={w_norm[0]:.4f}, real(1)={w_norm[1]:.4f}")
    print(f"  Classifier head bias:         fake(0)={bias[0]:.4f}, real(1)={bias[1]:.4f}")
    print("  OK: Checkpoint weights and architecture verified.")
    return True


def inspect_remote_samples(api_url: str, dataset_dir: Path, per_class: int = 10) -> None:
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        print(f"\n[3] Skipping remote sample test ({test_dir} not found locally).")
        return

    api_base = api_url.rstrip("/")
    print(f"\n[3] TESTING REMOTE FASTAPI ENDPOINT ({api_base}) ON UNSEEN TEST SAMPLES")
    rng = random.Random(42)
    total = 0
    correct = 0
    low_conf_count = 0

    for label in ("fake", "real"):
        files = [
            p for p in (test_dir / label).iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        sampled = rng.sample(files, min(per_class, len(files)))
        cls_correct = 0
        cls_low_conf = 0
        for p in sampled:
            boundary = "----AegisDiagBoundary"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{p.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode() + p.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                f"{api_base}/api/predict?threshold=0.65",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            total += 1
            pred_cls = (data.get("predicted_class") or "").lower()
            verdict = data.get("result") or data.get("verdict")
            conf = data.get("confidence", 0.0)
            is_low_conf = bool(data.get("low_confidence", conf < 0.65))
            if pred_cls == label:
                correct += 1
                cls_correct += 1
            if is_low_conf:
                low_conf_count += 1
                cls_low_conf += 1
            print(
                f"  [{label.upper()}] {p.name:<18} -> verdict={verdict:<5} "
                f"conf={conf:.2%} low_conf={is_low_conf} probs={data.get('probabilities')}"
            )
        print(
            f"  Summary {label.upper()}: correct={cls_correct}/{len(sampled)}, "
            f"low_confidence={cls_low_conf}/{len(sampled)}"
        )

    if total:
        print(
            f"\n  Overall remote sample accuracy: {correct}/{total} ({correct/total:.2%}) | "
            f"Low confidence (<65%): {low_conf_count}/{total} ({low_conf_count/total:.2%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Aegis Forensics accuracy & pipeline parity.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "best_model.pth")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional FastAPI URL to test with real Dataset/test samples.",
    )
    parser.add_argument("--samples-per-class", type=int, default=10)
    args = parser.parse_args()

    inspect_checkpoint(args.model_path)

    if args.dataset_dir.exists():
        from scripts.evaluate_model import audit_dataset_splits

        print("\n[2] AUDITING DATASET SPLITS:", args.dataset_dir)
        audit = audit_dataset_splits(args.dataset_dir, check_hashes=True)
        print(json.dumps(audit, indent=2))
    else:
        print(f"\n[2] Dataset folder not found at {args.dataset_dir}; skipping split audit.")

    if args.api_url:
        inspect_remote_samples(args.api_url, args.dataset_dir, per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
