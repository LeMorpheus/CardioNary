"""Deterministic image corruptions for chest X-ray robustness evaluation."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


SEVERITIES = {
    "blur": [0.5, 1.0, 1.5, 2.0, 3.0],
    "noise": [0.02, 0.04, 0.06, 0.08, 0.10],
    "contrast": [0.75, 0.60, 0.45, 0.30, 0.20],
    "jpeg": [80, 60, 40, 20, 10],
}


def apply_corruption(
    image: np.ndarray,
    corruption: str,
    severity: int,
    seed: int = 0,
) -> np.ndarray:
    """Apply one corruption to a grayscale image in [0, 1].

    Args:
        image: 2D grayscale image, normally 96x96, with values in [0, 1].
        corruption: One of blur, noise, contrast, jpeg.
        severity: Integer from 1 to 5.
        seed: Random seed used for Gaussian noise.

    Returns:
        Corrupted float32 image with values clipped to [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Image must be non-empty and contain finite values")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Image values must be in [0, 1]")
    if corruption not in SEVERITIES:
        raise ValueError(f"Unknown corruption: {corruption}")
    if not isinstance(severity, (int, np.integer)) or not 1 <= severity <= 5:
        raise ValueError("Severity must be an integer from 1 to 5")

    level = SEVERITIES[corruption][severity - 1]

    if corruption == "blur":
        result = gaussian_filter(image, sigma=level)

    elif corruption == "noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, level, size=image.shape)
        result = image + noise

    elif corruption == "contrast":
        mean = float(image.mean())
        result = (image - mean) * level + mean

    else:  # JPEG compression
        pixels = np.round(image * 255).astype(np.uint8)
        buffer = BytesIO()

        Image.fromarray(pixels).save(
            buffer,
            format="JPEG",
            quality=level,
        )

        buffer.seek(0)
        with Image.open(buffer) as compressed:
            result = np.asarray(
                compressed.convert("L"),
                dtype=np.float32,
            ) / 255.0

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def generate_corruptions(
    image: np.ndarray,
    seed: int = 0,
) -> dict[str, dict[int, np.ndarray]]:
    """Generate all 20 corruption/severity combinations for one image."""
    return {
        corruption: {
            severity: apply_corruption(image, corruption, severity, seed)
            for severity in range(1, 6)
        }
        for corruption in SEVERITIES
    }
