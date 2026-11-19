"""Model loading and lifecycle wrapper for the trained EfficientNet-B0 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from backend.config import (
    DEFAULT_FAKE_DECISION_THRESHOLD,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    EXPECTED_CLASS_MAPPING,
    IMAGE_SIZE,
)
from backend.predict import run_prediction
from backend.preprocessing import ImageValidationError, build_eval_transform

logger = logging.getLogger("aegis.model_loader")

__all__ = [
    "DeepfakeDetector",
    "ImageValidationError",
    "ModelMetadata",
]


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    device: str
    image_size: int
    class_to_idx: dict[str, int]
    total_parameters: int
    trainable_parameters: int
    low_confidence_threshold: float
    fake_decision_threshold: float


class DeepfakeDetector:
    """Loads the trained EfficientNet-B0 checkpoint once at startup and serves predictions."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        fake_decision_threshold: float = DEFAULT_FAKE_DECISION_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not 0.5 <= low_confidence_threshold < 1.0:
            raise ValueError("low_confidence_threshold must be in [0.5, 1.0).")
        if not 0.05 <= fake_decision_threshold <= 0.95:
            raise ValueError("fake_decision_threshold must be in [0.05, 0.95].")
        self.low_confidence_threshold = low_confidence_threshold
        self.fake_decision_threshold = fake_decision_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None
        self.image_size: int = IMAGE_SIZE
        self.class_to_idx: dict[str, int] = dict(EXPECTED_CLASS_MAPPING)
        self.transform = None
        self.metadata: ModelMetadata | None = None
        self.checkpoint_sha256: str = ""
        self.checkpoint_size_bytes: int = 0
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.transform is not None

    def load(self) -> None:
        """Load weights from models/best_model.pth once at startup."""
        if not self.model_path.exists():
            self.load_error = (
                f"Trained checkpoint not found at '{self.model_path}'. "
                "Place your existing models/best_model.pth file at this location."
            )
            logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
            return

        ckpt_bytes = self.model_path.read_bytes()
        self.checkpoint_size_bytes = len(ckpt_bytes)
        self.checkpoint_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as exc:
            self.load_error = f"Failed to load checkpoint '{self.model_path}': {exc}"
            logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
            return

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            self.load_error = (
                f"Checkpoint '{self.model_path}' is missing 'model_state_dict'."
            )
            logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
            return

        class_to_idx = checkpoint.get("class_to_idx")
        if class_to_idx != EXPECTED_CLASS_MAPPING:
            self.load_error = (
                f"Unexpected class_to_idx in checkpoint: {class_to_idx}. "
                f"Expected {EXPECTED_CLASS_MAPPING}."
            )
            logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
            return

        image_size = int(checkpoint.get("image_size", IMAGE_SIZE))

        model = models.efficientnet_b0(weights=None)
        input_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(input_features, 2)

        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as exc:
            self.load_error = (
                f"State dict architecture mismatch for EfficientNet-B0: {exc}"
            )
            logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
            return

        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                self.load_error = (
                    f"Checkpoint '{self.model_path}' contains NaN/Inf weights in '{name}'."
                )
                logger.error("[AEGIS DIAGNOSTIC] %s", self.load_error)
                return

        model.to(self.device)
        model.eval()

        self.model = model
        self.image_size = image_size
        self.class_to_idx = dict(class_to_idx)
        self.transform = build_eval_transform(self.image_size, preserve_aspect_ratio=False)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.metadata = ModelMetadata(
            architecture="EfficientNet-B0",
            checkpoint_path=str(self.model_path),
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_size_bytes=self.checkpoint_size_bytes,
            device=str(self.device),
            image_size=self.image_size,
            class_to_idx=self.class_to_idx,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            low_confidence_threshold=self.low_confidence_threshold,
            fake_decision_threshold=self.fake_decision_threshold,
        )
        self.load_error = None

        w = model.classifier[1].weight.detach()
        b = model.classifier[1].bias.detach()
        logger.info(
            "[AEGIS DIAGNOSTIC] Model loaded | checkpoint=%s | sha256=%s | size=%d bytes | "
            "device=%s | image_size=%d | class_to_idx=%s | classifier=%s | "
            "head_norm=(fake:%.4f, real:%.4f) | head_bias=(fake:%.4f, real:%.4f)",
            self.model_path,
            self.checkpoint_sha256[:16],
            self.checkpoint_size_bytes,
            self.device,
            self.image_size,
            self.class_to_idx,
            model.classifier[1],
            w[0].norm().item(),
            w[1].norm().item(),
            b[0].item(),
            b[1].item(),
        )

    def predict_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload",
        threshold: float | None = None,
        fake_decision_threshold: float | None = None,
        include_gradcam: bool = False,
    ) -> dict[str, Any]:
        """Run validation, preprocessing, inference, and optional Grad-CAM."""
        if not self.is_loaded or self.model is None or self.transform is None:
            raise RuntimeError(self.load_error or "Model checkpoint is not loaded.")

        return run_prediction(
            model=self.model,
            transform=self.transform,
            device=self.device,
            class_to_idx=self.class_to_idx,
            checkpoint_name=str(self.model_path.name),
            checkpoint_path=str(self.model_path),
            checkpoint_sha256=self.checkpoint_sha256,
            image_size=self.image_size,
            default_threshold=self.low_confidence_threshold,
            image_bytes=image_bytes,
            filename=filename,
            threshold=threshold,
            fake_decision_threshold=fake_decision_threshold,
            default_fake_decision_threshold=self.fake_decision_threshold,
            include_gradcam=include_gradcam,
        )
