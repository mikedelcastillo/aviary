#!/usr/bin/env python3
"""Generate cut-paste IR composites: IR-toned day-bambi crops pasted into REAL
train-split night-IR background frames, labelled budgie (class 8).

Train-split sources only (so val/test stay real for an honest A/B). Each composite
keeps the background frame's own labels and adds the pasted budgie box.

  PYTHONPATH=training uv run --no-sync python training/scripts/cut_paste_ir.py \
    --dataset-dir data/training/datasets/live --output-dir data/training/_cutpaste_stage
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import numpy as np

from aviary_training.cut_paste_ir import paste_object, yolo_line

BAMBI = 5
BUDGIE = 8
IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/training/datasets/live"))
    p.add_argument("--output-dir", type=Path, default=Path("data/training/_cutpaste_stage"))
    p.add_argument("--count", type=int, default=0, help="Number of composites (0 = one per IR background)")
    p.add_argument("--min-width", type=float, default=0.05, help="Min pasted budgie width (frac of frame)")
    p.add_argument("--max-width", type=float, default=0.16, help="Max pasted budgie width (frac of frame)")
    p.add_argument("--prefix", default="cutpaste__")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _img_for(label: Path, img_dir: Path) -> Path | None:
    return next((img_dir / f"{label.stem}{e}" for e in IMG_EXTS if (img_dir / f"{label.stem}{e}").exists()), None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ds = args.dataset_dir.resolve()
    img_train, lbl_train = ds / "images" / "train", ds / "labels" / "train"

    # Day-bambi source crops (RGB, cropped to the bambi bbox).
    crops: list[Image.Image] = []
    for txt in sorted(lbl_train.glob("*_day_*.txt")):
        img = _img_for(txt, img_train)
        if img is None:
            continue
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5 and p[0] == str(BAMBI):
                cx, cy, w, h = map(float, p[1:5])
                with Image.open(img) as im:
                    W, Hh = im.size
                    l, t = int((cx - w / 2) * W), int((cy - h / 2) * Hh)
                    r, b = int((cx + w / 2) * W), int((cy + h / 2) * Hh)
                    if r - l >= 8 and b - t >= 8:
                        crops.append(im.convert("RGB").crop((max(0, l), max(0, t), min(W, r), min(Hh, b))))

    # Real IR backgrounds (image + its label lines), train split only.
    backgrounds: list[tuple[Path, list[str]]] = []
    for txt in sorted(lbl_train.glob("*_ir_*.txt")):
        img = _img_for(txt, img_train)
        if img is not None:
            backgrounds.append((img, txt.read_text(encoding="utf-8").splitlines()))

    if not crops or not backgrounds:
        raise SystemExit(f"Need both bambi crops ({len(crops)}) and IR backgrounds ({len(backgrounds)}).")

    out_img, out_lbl = args.output_dir / "images", args.output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n = args.count or len(backgrounds)
    made = 0
    for i in range(n):
        bg_path, bg_lines = backgrounds[i % len(backgrounds)]
        crop = crops[int(rng.integers(len(crops)))]
        cx, cy = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
        wf = float(rng.uniform(args.min_width, args.max_width))
        with Image.open(bg_path) as bg:
            comp, box = paste_object(bg, crop, cx, cy, wf)
        stem = f"{args.prefix}{i:04d}_{bg_path.stem}"
        comp.save(out_img / f"{stem}.jpg", quality=90)
        lines = [ln for ln in bg_lines if ln.strip()] + [yolo_line(BUDGIE, box)]
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        made += 1

    print(f"Generated {made} cut-paste IR composites from {len(crops)} bambi crops x {len(backgrounds)} IR bgs -> {args.output_dir}")


if __name__ == "__main__":
    main()
