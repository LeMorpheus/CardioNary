"""Image front-end for the Chiron pulmonary (chest X-ray) model.

Mirrors core/features.py in shape and intent: one place that defines exactly what
the model sees, written so the identical operations can run on the MCU.

On-device the pipeline is trivial compared with the audio path -- decode is done
offline, so the firmware only holds a pre-resized uint8 image and applies a
scale/offset into int8. That is the whole point of choosing 96x96 grayscale:
the "DSP" stage has no FFT, no filterbank, and therefore no fixed-point parity
risk (docs/06_MODEL_DESIGN.md section 2.2 does not apply here).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageConfig:
    """Image front-end parameters. Mirrored by chiron_img_config.h on device."""

    name: str = "pulmo_cxr"
    size: int = 96                 # 96x96 grayscale -> 9,216 bytes per image
    grayscale: bool = True
    # Chest films have wide, inconsistent aspect ratios (384x127 to 2916x2713 in
    # this corpus). Centre-cropping to square before resize keeps the lung
    # fields and avoids the anatomical distortion that plain resizing causes.
    center_crop: bool = True
    equalize: bool = False         # histogram equalisation, ablated in training

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["bytes_per_image"] = self.size * self.size
        return d


CXR_CONFIG = ImageConfig()


def load_image(path: str, cfg: ImageConfig = CXR_CONFIG) -> np.ndarray:
    """Load one chest film -> float32 [0,1], shape (size, size).

    Deterministic and dependency-light: PIL only, no augmentation, no random
    state. Whatever this returns is exactly what gets quantised.
    """
    with Image.open(path) as im:
        if cfg.grayscale:
            im = im.convert("L")

        if cfg.center_crop:
            w, h = im.size
            s = min(w, h)
            left, top = (w - s) // 2, (h - s) // 2
            im = im.crop((left, top, left + s, top + s))

        im = im.resize((cfg.size, cfg.size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)

    if cfg.equalize:
        # simple contrast stretch on the 1st/99th percentiles
        lo, hi = np.percentile(a, [1, 99])
        if hi > lo:
            a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0

    return a / 255.0


def features_for_image(path: str, cfg: ImageConfig = CXR_CONFIG) -> np.ndarray:
    """One image -> (1, size, size) stack, matching the audio API's shape."""
    return load_image(path, cfg)[None, ...].astype(np.float32)


if __name__ == "__main__":
    cfg = CXR_CONFIG
    print("Chiron chest X-ray front-end")
    for k, v in cfg.to_dict().items():
        print(f"  {k:18s} {v}")
