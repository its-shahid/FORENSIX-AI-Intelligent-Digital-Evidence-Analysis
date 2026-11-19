"""Central configuration for Aegis Forensics EfficientNet-B0 inference backend."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_path(raw_value: str) -> Path:
    candidate = Path(raw_value.strip())
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


DEFAULT_MODEL_PATH = _resolve_model_path(
    os.getenv("MODEL_PATH", "models/best_model.pth")
)
MODEL_CONFIG_PATH = ROOT / "models" / "model_config.json"
RESULTS_DIR = ROOT / "results"
METRICS_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
PRECISION_RECALL_PATH = RESULTS_DIR / "precision_recall_curve.png"

EXPECTED_CLASS_MAPPING = {"fake": 0, "real": 1}
IMAGE_SIZE = 224
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_LOW_CONFIDENCE_THRESHOLD = float(
    os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.65")
)
DEFAULT_FAKE_DECISION_THRESHOLD = float(
    os.getenv("FAKE_DECISION_THRESHOLD", "0.50")
)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

_raw_cors = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGINS: list[str] = (
    [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]
    if _raw_cors
    else list(DEFAULT_CORS_ORIGINS)
)
_extra_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
if _extra_origin and _extra_origin not in CORS_ORIGINS:
    CORS_ORIGINS.append(_extra_origin)

CORS_ORIGIN_REGEX: str | None = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https://([a-z0-9-]+\.)*run\.app$",
)
