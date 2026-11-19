"""Prediction and Grad-CAM explainability helpers for EfficientNet-B0."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.preprocessing import ImageValidationError, validate_and_decode_image

logger = logging.getLogger("aegis.predict")


def _apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map normalized [0, 1] activation map to an RGB jet-like heatmap without requiring matplotlib."""
    x = np.clip(cam, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def generate_gradcam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    rgb_image: Image.Image,
    class_idx: int,
    image_size: int = 224,
) -> str | None:
    """Compute Grad-CAM overlay on model.features[-1] (matching app_422c9ef3.py)."""
    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    features = getattr(model, "features", None)
    if features is None or len(features) == 0:
        return None

    target_layer = features[-1]

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    hook = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        grad_tensor = image_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            output = model(grad_tensor)
            score = output[:, class_idx]
            score.backward()

        if not gradients or not activations:
            return None

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        original = np.asarray(rgb_image.resize((image_size, image_size)), dtype=np.float32)
        heatmap = _apply_jet_colormap(cam_np).astype(np.float32)
        overlay = np.clip(0.5 * original + 0.5 * heatmap, 0, 255).astype(np.uint8)

        out_buf = BytesIO()
        Image.fromarray(overlay).save(out_buf, format="PNG")
        encoded = base64.b64encode(out_buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def classify_binary_probabilities(
    fake_prob: float,
    real_prob: float,
    low_confidence_threshold: float = 0.65,
    fake_decision_threshold: float = 0.50,
) -> dict[str, Any]:
    """Classify binary FAKE (0) / REAL (1) probabilities strictly as REAL or FAKE.

    Rule:
    1. If REAL probability > FAKE probability -> REAL
    2. If FAKE probability >= REAL probability -> FAKE
    Cutoff slider controls the low-confidence warning only.
    """
    is_fake = (
        fake_prob >= real_prob
        if abs(fake_decision_threshold - 0.5) < 1e-9
        else fake_prob >= fake_decision_threshold
    )
    verdict = "FAKE" if is_fake else "REAL"
    confidence = fake_prob if is_fake else real_prob
    low_confidence = confidence < low_confidence_threshold
    return {
        "verdict": verdict,
        "is_fake": is_fake,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "confidence_warning": (
            "Low confidence – prediction may be unreliable."
            if low_confidence
            else None
        ),
    }


def run_prediction(
    model: nn.Module,
    transform: Any,
    device: torch.device,
    class_to_idx: dict[str, int],
    checkpoint_name: str,
    image_size: int,
    default_threshold: float,
    image_bytes: bytes,
    filename: str = "upload",
    threshold: float | None = None,
    fake_decision_threshold: float | None = None,
    default_fake_decision_threshold: float = 0.50,
    include_gradcam: bool = False,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate, preprocess, run EfficientNet-B0 inference, and format JSON response."""
    active_threshold = default_threshold if threshold is None else float(threshold)
    if not 0.5 <= active_threshold < 1.0:
        raise ImageValidationError(
            "Low-confidence warning threshold must be between 0.50 and 0.99."
        )

    active_fake_threshold = (
        default_fake_decision_threshold
        if fake_decision_threshold is None
        else float(fake_decision_threshold)
    )
    if not 0.05 <= active_fake_threshold <= 0.95:
        raise ImageValidationError(
            "Fake decision threshold must be between 0.05 and 0.95."
        )

    rgb_image, image_meta = validate_and_decode_image(
        image_bytes=image_bytes,
        filename=filename,
    )

    model.eval()
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - start
    inference_time_ms = inference_time_sec * 1000.0

    fake_idx = class_to_idx["fake"]  # 0
    real_idx = class_to_idx["real"]  # 1

    fake_prob = float(probabilities[fake_idx].item())
    real_prob = float(probabilities[real_idx].item())
    fake_logit = float(logits[0, fake_idx].item())
    real_logit = float(logits[0, real_idx].item())

    decision = classify_binary_probabilities(
        fake_prob=fake_prob,
        real_prob=real_prob,
        low_confidence_threshold=active_threshold,
        fake_decision_threshold=active_fake_threshold,
    )
    is_fake = bool(decision["is_fake"])
    raw_label = str(decision["verdict"])
    predicted_class_idx = fake_idx if is_fake else real_idx
    confidence = float(decision["confidence"])
    margin = abs(real_prob - fake_prob)
    verdict = raw_label
    low_confidence = bool(decision["low_confidence"])
    confidence_warning = decision["confidence_warning"]

    logger.info(
        "[AEGIS DIAGNOSTIC] file=%s (%dx%d %s) | checkpoint=%s (sha256=%s) | "
        "device=%s | eval_mode=%s | tensor=[min:%.3f max:%.3f mean:%.3f std:%.3f] | "
        "logits=(fake[0]:%.4f, real[1]:%.4f) | probs=(fake:%.4f, real:%.4f) | "
        "verdict=%s (fake_thresh=%.3f, low_confidence=%s, warn_thresh=%.2f)",
        filename,
        image_meta["width"],
        image_meta["height"],
        image_meta["format"],
        checkpoint_path or checkpoint_name,
        (checkpoint_sha256 or "")[:16],
        device,
        not model.training,
        float(tensor.min().item()),
        float(tensor.max().item()),
        float(tensor.mean().item()),
        float(tensor.std().item()),
        fake_logit,
        real_logit,
        fake_prob,
        real_prob,
        verdict,
        active_fake_threshold,
        low_confidence,
        active_threshold,
    )

    gradcam_overlay: str | None = None
    if include_gradcam:
        gradcam_overlay = generate_gradcam_overlay(
            model=model,
            image_tensor=tensor,
            rgb_image=rgb_image,
            class_idx=predicted_class_idx,
            image_size=image_size,
        )

    return {
        "success": True,
        "result": verdict,
        "prediction": verdict,
        "verdict": verdict,
        "raw_label": raw_label,
        "predicted_class": raw_label.lower(),
        "class_id": predicted_class_idx,
        "confidence": round(confidence, 6),
        "confidence_percentage": round(confidence * 100.0, 2),
        "low_confidence": low_confidence,
        "confidence_warning": confidence_warning,
        "threshold": active_threshold,
        "fake_decision_threshold": active_fake_threshold,
        "margin": round(margin, 6),
        "inference_time": round(inference_time_sec, 4),
        "inference_time_ms": round(inference_time_ms, 2),
        "model": "EfficientNet-B0",
        "probabilities": {
            "fake": round(fake_prob, 6),
            "real": round(real_prob, 6),
        },
        "logits": {
            "fake": round(fake_logit, 6),
            "real": round(real_logit, 6),
        },
        "gradcam_overlay": gradcam_overlay,
        "image_metadata": image_meta,
        "model_info": {
            "architecture": "EfficientNet-B0",
            "checkpoint": checkpoint_name,
            "checkpoint_path": checkpoint_path or checkpoint_name,
            "checkpoint_sha256": (checkpoint_sha256 or "")[:16],
            "device": str(device),
            "image_size": image_size,
            "class_mapping": class_to_idx,
            "low_confidence_threshold": active_threshold,
            "fake_decision_threshold": active_fake_threshold,
        },
        "disclaimer": (
            "Probabilistic neural network inference from EfficientNet-B0. "
            "Predictions indicate statistical similarity to training classes "
            "and do not constitute definitive proof of authenticity or manipulation."
        ),
    }
