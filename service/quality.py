"""Cheap image statistics used twice: to measure how the two sites differ, and at serving time
to flag photos that look unlike the images the model was trained on. Pillow and NumPy only."""
from __future__ import annotations

import numpy as np
from PIL import Image

FEATURES = ("brightness", "contrast", "saturation", "sharpness", "red_share", "blue_share")


def features(img: Image.Image) -> dict:
    im = img.convert("RGB")
    im.thumbnail((256, 256))
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    mx, mn = a.max(-1), a.min(-1)
    total = a.sum(-1) + 1e-6
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()),
        "sharpness": float(lap.var() * 1000),
        "red_share": float((a[..., 0] / total).mean()),
        "blue_share": float((a[..., 2] / total).mean()),
    }


def outside_range(feats: dict, ranges: dict) -> list[str]:
    """Names of features that fall outside the training-site range stored with the model."""
    out = []
    for name, (lo, hi) in (ranges or {}).items():
        if name in feats and not (lo <= feats[name] <= hi):
            out.append(name)
    return out
