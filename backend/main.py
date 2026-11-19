"""FastAPI inference server for the trained EfficientNet-B0 deepfake detector."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.config import (
    CONFUSION_MATRIX_PATH,
    CORS_ORIGIN_REGEX,
    CORS_ORIGINS,
    DEFAULT_FAKE_DECISION_THRESHOLD,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    EXPECTED_CLASS_MAPPING,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_IMAGE_BYTES,
    METRICS_JSON_PATH,
    PRECISION_RECALL_PATH,
)
from backend.model_loader import DeepfakeDetector, ImageValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = DeepfakeDetector(
        model_path=DEFAULT_MODEL_PATH,
        low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        fake_decision_threshold=DEFAULT_FAKE_DECISION_THRESHOLD,
    )
    detector.load()
    app.state.detector = detector
    yield


app = FastAPI(
    title="Aegis Forensics Deepfake Detection API",
    description=(
        "Inference API wrapping the existing trained EfficientNet-B0 checkpoint "
        "(models/best_model.pth) with class mapping fake=0, real=1."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error during inference.",
            "message": str(exc),
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Aegis Forensics Deepfake Detection API",
        "health": "/api/health",
        "predict": "/api/predict",
        "performance": "/api/performance",
    }


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    detector: DeepfakeDetector = app.state.detector
    return {
        "status": "ok" if detector.is_loaded else "checkpoint_missing",
        "python_backend_reachable": True,
        "model_loaded": detector.is_loaded,
        "model_name": "EfficientNet-B0",
        "model_error": detector.load_error,
        "checkpoint_path": str(detector.model_path),
        "checkpoint_sha256": detector.checkpoint_sha256[:16] if detector.checkpoint_sha256 else None,
        "architecture": "EfficientNet-B0",
        "class_mapping": (
            detector.class_to_idx if detector.is_loaded else EXPECTED_CLASS_MAPPING
        ),
        "image_size": detector.image_size,
        "device": str(detector.device),
        "low_confidence_threshold": detector.low_confidence_threshold,
        "fake_decision_threshold": detector.fake_decision_threshold,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    detector: DeepfakeDetector = app.state.detector
    meta = detector.metadata
    return {
        "model_loaded": detector.is_loaded,
        "model_name": "EfficientNet-B0",
        "model_error": detector.load_error,
        "architecture": "EfficientNet-B0 (torchvision.models.efficientnet_b0)",
        "classifier_head": "nn.Linear(1280, 2)",
        "checkpoint_path": str(detector.model_path),
        "checkpoint_sha256": detector.checkpoint_sha256[:16] if detector.checkpoint_sha256 else None,
        "checkpoint_size_bytes": detector.checkpoint_size_bytes,
        "class_mapping": EXPECTED_CLASS_MAPPING,
        "preprocessing": {
            "resize": [detector.image_size, detector.image_size],
            "color_mode": "RGB",
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
        },
        "total_parameters": meta.total_parameters if meta else None,
        "device": str(detector.device),
        "low_confidence_threshold": detector.low_confidence_threshold,
        "fake_decision_threshold": detector.fake_decision_threshold,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float | None = Query(default=None, ge=0.5, le=0.99),
    fake_threshold: float | None = Query(default=None, ge=0.05, le=0.95),
    include_gradcam: bool = Query(default=False),
) -> JSONResponse:
    detector: DeepfakeDetector = app.state.detector
    if not detector.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Model checkpoint is not loaded.",
                "message": detector.load_error,
                "checkpoint_path": str(detector.model_path),
            },
        )

    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "success": False,
                    "error": "Uploaded image is too large.",
                    "message": (
                        f"Image exceeds maximum allowed size of "
                        f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
                    ),
                },
            )
        chunks.append(chunk)
    payload = b"".join(chunks)

    try:
        result = detector.predict_bytes(
            image_bytes=payload,
            filename=file.filename or "uploaded_image",
            threshold=threshold,
            fake_decision_threshold=fake_threshold,
            include_gradcam=include_gradcam,
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "Invalid image upload.",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Inference failed.",
                "message": str(exc),
            },
        ) from exc

    return JSONResponse(content=result)


@app.get("/api/performance")
def performance() -> dict[str, Any]:
    has_cm_image = CONFUSION_MATRIX_PATH.exists()
    has_pr_image = PRECISION_RECALL_PATH.exists()

    if not METRICS_JSON_PATH.exists():
        return {
            "evaluated": False,
            "confusion_matrix_image_available": has_cm_image,
            "confusion_matrix_image_url": (
                "/api/results/confusion_matrix.png" if has_cm_image else None
            ),
            "precision_recall_image_available": has_pr_image,
            "precision_recall_image_url": (
                "/api/results/precision_recall_curve.png" if has_pr_image else None
            ),
            "expected_metrics_path": str(METRICS_JSON_PATH),
            "message": (
                "No saved evaluation metrics found yet. Run "
                "`python scripts/evaluate_model.py` against models/best_model.pth "
                "and Dataset/test to generate actual test metrics."
            ),
        }

    try:
        with METRICS_JSON_PATH.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to read evaluation metrics JSON.",
                "message": str(exc),
            },
        ) from exc

    metrics["evaluated"] = True
    metrics["confusion_matrix_image_available"] = has_cm_image
    metrics["confusion_matrix_image_url"] = (
        "/api/results/confusion_matrix.png" if has_cm_image else None
    )
    metrics["precision_recall_image_available"] = has_pr_image
    metrics["precision_recall_image_url"] = (
        "/api/results/precision_recall_curve.png" if has_pr_image else None
    )
    return metrics


@app.get("/api/results/confusion_matrix.png")
def confusion_matrix_image() -> FileResponse:
    if not CONFUSION_MATRIX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="results/confusion_matrix.png has not been generated yet.",
        )
    return FileResponse(CONFUSION_MATRIX_PATH, media_type="image/png")


@app.get("/api/results/precision_recall_curve.png")
def precision_recall_image() -> FileResponse:
    if not PRECISION_RECALL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="results/precision_recall_curve.png has not been generated yet.",
        )
    return FileResponse(PRECISION_RECALL_PATH, media_type="image/png")
