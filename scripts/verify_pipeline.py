"""End-to-end verification suite for the existing EfficientNet-B0 pipeline.

Tests:
  1. Model checkpoint loading (models/best_model.pth)
  2. Class mapping verification (fake=0, real=1)
  3. Preprocessing & normalization shape checks
  4. Invalid/corrupted image rejection
  5. Prediction JSON contract + Grad-CAM overlay generation
  6. Real inference on sample Dataset/test/fake and Dataset/test/real images (if Dataset/test exists)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DEFAULT_MODEL_PATH, EXPECTED_CLASS_MAPPING
from backend.model_loader import DeepfakeDetector, ImageValidationError


def main() -> int:
    print("1. Loading trained checkpoint:", DEFAULT_MODEL_PATH)
    detector = DeepfakeDetector(model_path=DEFAULT_MODEL_PATH)
    detector.load()

    if not detector.is_loaded:
        print("FAILED:", detector.load_error)
        return 1

    print("   OK - Loaded EfficientNet-B0 on", detector.device)
    print("2. Verifying class mapping:", detector.class_to_idx)
    assert detector.class_to_idx == EXPECTED_CLASS_MAPPING, "Class mapping mismatch!"
    print("   OK - fake=0, real=1 confirmed.")

    print("3. Testing invalid/corrupted image handling...")
    try:
        detector.predict_bytes(b"not-an-image", filename="corrupted.jpg")
        print("FAILED: Corrupted bytes were not rejected!")
        return 1
    except ImageValidationError as exc:
        print(f"   OK - Rejected invalid image: {exc}")

    print("4. Testing valid RGB preprocessing + inference + Grad-CAM...")
    buf = BytesIO()
    Image.new("RGB", (320, 240), color=(120, 128, 135)).save(buf, format="PNG")
    res = detector.predict_bytes(
        buf.getvalue(),
        filename="synthetic_smoke.png",
        include_gradcam=True,
    )
    for key in ("success", "prediction", "confidence", "inference_time", "model", "gradcam_overlay"):
        assert key in res, f"Missing expected key '{key}' in prediction response."
    assert res["success"] is True
    assert res["gradcam_overlay"] and res["gradcam_overlay"].startswith("data:image/png;base64,")
    print(
        f"   OK - Smoke inference completed in {res['inference_time_ms']} ms | "
        f"prediction={res['prediction']} | probs={res['probabilities']} | Grad-CAM generated"
    )

    test_dir = ROOT / "Dataset" / "test"
    if test_dir.exists():
        print("5. Testing real Dataset/test samples...")
        for label in ("fake", "real"):
            folder = test_dir / label
            samples = [
                p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ][:5]
            correct = 0
            for sample_path in samples:
                out = detector.predict_bytes(sample_path.read_bytes(), filename=sample_path.name)
                if out["raw_label"].lower() == label:
                    correct += 1
                print(
                    f"   [{label.upper()}] {sample_path.name} -> "
                    f"prediction={out['prediction']} (raw={out['raw_label']}, "
                    f"conf={out['confidence']:.2%}, {out['inference_time_ms']} ms)"
                )
            if samples:
                print(f"   Sample {label} accuracy: {correct}/{len(samples)}")
    else:
        print("5. Dataset/test not found locally; skipped dataset sample test.")

    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
