"""Image validation, RGB conversion, and training-parity tensor preprocessing."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from backend.config import (
    ALLOWED_FORMATS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_IMAGE_BYTES,
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails format or integrity validation."""


def center_crop_square(image: Image.Image) -> Image.Image:
    """Crop rectangular images to a centered square without altering 1:1 dataset images."""
    width, height = image.size
    if width == height:
        return image
    side = min(width, height)
    left = (width - side) // 2
    # Slightly favor upper-center for tall portrait photos where faces sit above midpoint
    top = (height - side) // 3 if height > width else (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def build_eval_transform(
    image_size: int = 224,
    preserve_aspect_ratio: bool = False,
) -> transforms.Compose:
    """Return the exact preprocessing transform used during training & diagnostic script."""
    steps: list[Any] = []
    if preserve_aspect_ratio:
        steps.append(transforms.Lambda(center_crop_square))
    steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def validate_and_decode_image(
    image_bytes: bytes,
    filename: str = "upload",
) -> tuple[Image.Image, dict[str, Any]]:
    """Validate uploaded image bytes, fix EXIF orientation, and decode to RGB."""
    if not image_bytes:
        raise ImageValidationError("Uploaded file is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            detected_format = (probe.format or "").upper()
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            f"Invalid or corrupted image file '{filename}': {exc}"
        ) from exc

    if detected_format not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format '{detected_format or 'unknown'}'. "
            f"Supported formats: {', '.join(sorted(ALLOWED_FORMATS))}."
        )

    with Image.open(BytesIO(image_bytes)) as img:
        transposed = ImageOps.exif_transpose(img)
        orig_width, orig_height = transposed.size
        orig_mode = transposed.mode
        rgb_image = transposed.convert("RGB")

    if orig_width <= 0 or orig_height <= 0:
        raise ImageValidationError("Image has invalid dimensions.")

    metadata = {
        "filename": filename,
        "format": detected_format,
        "width": orig_width,
        "height": orig_height,
        "mode": orig_mode,
        "size_bytes": len(image_bytes),
    }
    return rgb_image, metadata
