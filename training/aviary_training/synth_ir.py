"""Synthesize IR-style training frames from daytime RGB frames.

Motivation: the IR/night ``budgie`` species class is the one genuinely
data-starved class, and ``bambi`` (the sole budgie) is photographed plentifully
in DAY frames. Converting day frames to an IR-realistic look and relabelling the
named individuals to their species lets those day frames augment the starved IR
classes — chiefly ``budgie`` — without waiting for night capture.

Pure logic only (numpy/PIL, no torch), so the transform + label remap are unit
testable. Real night IR is monochrome (R=G=B), IR-LED lit (center-bright, blown
highlights), contrasty and grainy; ``to_ir`` approximates that. It is NOT a
perfect physical model — it is a silhouette/appearance proxy whose value is
decided empirically by the budgie-recall A/B, not by fidelity alone.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Individual (DAY) roster index -> species (IR) roster index. Global roster order:
# 0 draft, 1 pizza -> 6 cockatiel; 2 percy, 3 matcha, 4 jynx -> 7 lovebird;
# 5 bambi -> 8 budgie. Species/unknown indices map to themselves.
DAY_TO_SPECIES: dict[int, int] = {0: 6, 1: 6, 2: 7, 3: 7, 4: 7, 5: 8}


def to_ir(
    img: Image.Image,
    *,
    seed: int = 0,
    gamma: float = 1.0,
    vignette: float = 0.22,
    contrast: float = 1.35,
    noise: float = 7.0,
    blur: float = 0.5,
    target_mean: float = 90.0,
) -> Image.Image:
    """RGB frame -> IR-style 3-channel grayscale (R=G=B), brightness-matched to
    real night IR (~90 mean). Deterministic for a given ``seed``."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    # Red-weighted grayscale: the red end carries more of the near-IR a Tapo
    # night camera actually senses than a perceptual (green-heavy) luma would.
    gray = (0.5 * a[..., 0] + 0.35 * a[..., 1] + 0.15 * a[..., 2]) / 255.0
    gray = np.power(np.clip(gray, 0.0, 1.0), gamma)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    gray = gray * (1.0 - vignette * np.clip(r, 0.0, 1.0) ** 2)  # mild IR-LED falloff
    gray = np.clip(0.5 + (gray - 0.5) * contrast, 0.0, 1.0)      # IR contrast
    g = gray * 255.0
    g *= target_mean / max(float(g.mean()), 1.0)                # match real-IR brightness
    rng = np.random.default_rng(seed)
    g = np.clip(g + rng.normal(0.0, noise, g.shape), 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(g, mode="L").convert("RGB")
    return out.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else out


def remap_to_species(lines: list[str]) -> list[str]:
    """Remap YOLO label lines, individual -> species (DAY_TO_SPECIES); keep
    geometry verbatim. Species/unknown lines pass through unchanged; malformed
    (<5 fields) lines are dropped."""
    out: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        cls = DAY_TO_SPECIES.get(cls, cls)
        out.append(" ".join([str(cls), *parts[1:]]))
    return out


def has_class(lines: list[str], cls: int) -> bool:
    """True if any YOLO line is for class ``cls``."""
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] == str(cls):
            return True
    return False
