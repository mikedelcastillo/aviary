"""Cut-and-paste IR augmentation: paste IR-toned day-bird crops into REAL night-IR
background frames.

Rationale (see the deep-research verdict): whole-scene RGB->NIR is ill-posed, so
naive grayscale hurt. Instead we keep the REAL night-IR background — which already
carries the correct active-IR lighting, hotspots, noise and close-up framing — and
only paste an IR-toned object crop into it. The model then learns the budgie's
appearance in a genuinely in-domain scene. Copy-paste is proven to lift downstream
detection, especially for rare classes (budgie here).

Pure logic (numpy/PIL, no torch) so the geometry/label math is unit-testable. The
object crop is a bbox (bird + some surrounding background); a feathered elliptical
alpha focuses the paste on the central bird and fades the rectangular edges so the
day-background bleed and hard seams are suppressed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def ir_tone(crop: Image.Image) -> np.ndarray:
    """RGB crop -> 2D float (0..255) red-weighted grayscale (near-IR proxy)."""
    a = np.asarray(crop.convert("RGB"), dtype=np.float32)
    return 0.5 * a[..., 0] + 0.35 * a[..., 1] + 0.15 * a[..., 2]


def feather_mask(w: int, h: int, edge: float = 0.35) -> np.ndarray:
    """Elliptical alpha (1 at center, fading to 0 by the rectangle corner) so the
    paste blends and the bbox-corner background fades out. ``edge`` = width of the
    fade band as a fraction of the radius."""
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - (w - 1) / 2) / ((w - 1) / 2 + 1e-6)
    ny = (yy - (h - 1) / 2) / ((h - 1) / 2 + 1e-6)
    r = np.sqrt(nx**2 + ny**2)  # 0 center -> ~1 at edge midpoints
    return np.clip((1.0 - r) / edge, 0.0, 1.0).astype(np.float32)


def paste_object(
    bg: Image.Image,
    crop: Image.Image,
    cx: float,
    cy: float,
    width_frac: float,
    *,
    box_shrink: float = 0.82,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Paste an IR-toned ``crop`` onto night-IR ``bg`` centered at normalized
    (cx,cy) with width = ``width_frac`` of the bg width (aspect preserved), brightness
    matched to the local background patch, with a feathered alpha. Returns the
    composite (RGB, grayscale-valued) and the normalized YOLO box (cx,cy,w,h) for
    the pasted object (shrunk by ``box_shrink`` toward the bird, since the feather
    fades the rectangle's background edges)."""
    arr = np.asarray(bg.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]
    cw, ch = crop.size
    tw = max(8, int(round(width_frac * W)))
    th = max(8, int(round(tw * ch / max(1, cw))))
    tw, th = min(tw, W), min(th, H)

    g = ir_tone(crop.resize((tw, th)))
    x0 = int(round(cx * W - tw / 2))
    y0 = int(round(cy * H - th / 2))
    x0 = max(0, min(W - tw, x0))
    y0 = max(0, min(H - th, y0))

    region = arr[y0 : y0 + th, x0 : x0 + tw, :]
    # Match the crop's brightness to the local IR patch so the bird sits in-scene.
    g = np.clip(g * (float(region.mean()) / (float(g.mean()) + 1e-6)), 0.0, 255.0)
    g3 = np.repeat(g[..., None], 3, axis=2)
    m3 = np.repeat(feather_mask(tw, th)[..., None], 3, axis=2)
    arr[y0 : y0 + th, x0 : x0 + tw, :] = m3 * g3 + (1.0 - m3) * region

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    box = ((x0 + tw / 2) / W, (y0 + th / 2) / H, (tw / W) * box_shrink, (th / H) * box_shrink)
    return out, box


def yolo_line(cls: int, box: tuple[float, float, float, float]) -> str:
    cx, cy, w, h = box
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
