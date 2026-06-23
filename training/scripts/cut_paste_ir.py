#!/usr/bin/env python3
"""Generate species-balanced cut-paste IR composites: IR-toned day-bird crops
pasted into REAL train-split night-IR backgrounds, labelled by SPECIES.

Refines the first pass (which lifted budgie +10 but regressed cockatiel -10):
  - pastes ALL species (cockatiel/lovebird/budgie), weighted toward the starved
    class (budgie) instead of budgie-only, so the IR class balance isn't skewed;
  - places each paste NON-OVERLAPPING with the background's real boxes, so a
    pasted bird never corrupts a real labelled bird.

Train-split sources only (val/test stay 100% real for an honest A/B). Each
composite keeps the background frame's own labels and adds the pasted box.

  PYTHONPATH=training python training/scripts/cut_paste_ir.py \
    --dataset-dir data/training/datasets/live --output-dir data/training/_cutpaste_stage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from aviary_training.cut_paste_ir import DAY_TO_SPECIES, overlaps_any, parse_boxes, paste_object, yolo_line

SPECIES = {6: "cockatiel", 7: "lovebird", 8: "budgie"}
IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/training/datasets/live"))
    p.add_argument("--output-dir", type=Path, default=Path("data/training/_cutpaste_stage"))
    p.add_argument("--count", type=int, default=300, help="Total composites to generate")
    # Sampling weights for budgie/lovebird/cockatiel — default boosts the starved budgie.
    p.add_argument("--w-budgie", type=float, default=0.55)
    p.add_argument("--w-lovebird", type=float, default=0.30)
    p.add_argument("--w-cockatiel", type=float, default=0.15)
    p.add_argument("--min-width", type=float, default=0.05, help="Min pasted width (frac of frame)")
    p.add_argument("--max-width", type=float, default=0.16, help="Max pasted width (frac of frame)")
    p.add_argument("--overlap-iou", type=float, default=0.05, help="Reject placements above this IoU with a real box")
    p.add_argument("--prefix", default="cutpaste__")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _img_for(label: Path, img_dir: Path) -> Path | None:
    return next((img_dir / f"{label.stem}{e}" for e in IMG_EXTS if (img_dir / f"{label.stem}{e}").exists()), None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ds = args.dataset_dir.resolve()
    img_train, lbl_train = ds / "images" / "train", ds / "labels" / "train"

    # Build per-species crop pools from DAY frames (individual -> species).
    pools: dict[int, list[Image.Image]] = {6: [], 7: [], 8: []}
    for txt in sorted(lbl_train.glob("*_day_*.txt")):
        img = _img_for(txt, img_train)
        if img is None:
            continue
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5 or not p[0].isdigit():
                continue
            sp = DAY_TO_SPECIES.get(int(p[0]))
            if sp is None:
                continue
            cx, cy, w, h = map(float, p[1:5])
            with Image.open(img) as im:
                W, Hh = im.size
                l, t, r, b = int((cx - w / 2) * W), int((cy - h / 2) * Hh), int((cx + w / 2) * W), int((cy + h / 2) * Hh)
                if r - l >= 8 and b - t >= 8:
                    pools[sp].append(im.convert("RGB").crop((max(0, l), max(0, t), min(W, r), min(Hh, b))))

    # Real IR backgrounds + their existing boxes (train split only).
    backgrounds: list[tuple[Path, list[str], list]] = []
    for txt in sorted(lbl_train.glob("*_ir_*.txt")):
        img = _img_for(txt, img_train)
        if img is not None:
            lines = txt.read_text(encoding="utf-8").splitlines()
            backgrounds.append((img, [ln for ln in lines if ln.strip()], parse_boxes(lines)))

    species = [6, 7, 8]
    weights = np.array([args.w_cockatiel, args.w_lovebird, args.w_budgie], dtype=float)
    # Drop species with no available crops, renormalize.
    avail = np.array([len(pools[s]) > 0 for s in species])
    if not avail.any() or not backgrounds:
        raise SystemExit(f"Need crops (cockatiel={len(pools[6])} lovebird={len(pools[7])} budgie={len(pools[8])}) and IR bgs ({len(backgrounds)}).")
    weights = np.where(avail, weights, 0.0)
    weights /= weights.sum()

    out_img, out_lbl = args.output_dir / "images", args.output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    made = {6: 0, 7: 0, 8: 0}
    for i in range(args.count):
        sp = species[int(rng.choice(3, p=weights))]
        crop = pools[sp][int(rng.integers(len(pools[sp])))]
        bg_path, bg_lines, bg_boxes = backgrounds[i % len(backgrounds)]
        wf = float(rng.uniform(args.min_width, args.max_width))
        # Find a non-overlapping placement (retry up to 25x), else best-effort.
        cx = cy = 0.5
        box = None
        for _ in range(25):
            cx, cy = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
            cand = (cx, cy, wf * 0.82, (wf * crop.size[1] / max(1, crop.size[0])) * 0.82)
            if not overlaps_any(cand, bg_boxes, args.overlap_iou):
                box = cand
                break
        if box is None:
            continue  # couldn't place without hitting a real bird; skip this one
        with Image.open(bg_path) as bg:
            comp, placed = paste_object(bg, crop, cx, cy, wf)
        stem = f"{args.prefix}{i:04d}_{SPECIES[sp]}_{bg_path.stem}"
        comp.save(out_img / f"{stem}.jpg", quality=90)
        (out_lbl / f"{stem}.txt").write_text("\n".join(bg_lines + [yolo_line(sp, placed)]) + "\n", encoding="utf-8")
        made[sp] += 1

    print(f"Generated {sum(made.values())} composites -> {args.output_dir} "
          f"(budgie={made[8]} lovebird={made[7]} cockatiel={made[6]}); "
          f"pools: budgie={len(pools[8])} lovebird={len(pools[7])} cockatiel={len(pools[6])}, bgs={len(backgrounds)}")


if __name__ == "__main__":
    main()
