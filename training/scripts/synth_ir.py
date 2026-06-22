#!/usr/bin/env python3
"""Generate IR-style synthetic training frames from train-split day frames.

Sources ONLY the dataset's TRAIN split (so val/test stay 100% real and the
budgie A/B is honest), keeps day frames that contain a chosen source class
(default ``bambi``=5, the sole budgie), renders each to an IR look, and writes a
synthetic image + species-remapped label into the train split (or a staging dir).

  PYTHONPATH=training uv run --no-sync python training/scripts/synth_ir.py \
    --dataset-dir data/training/datasets/live --output-dir <stage-or-train>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from aviary_training.synth_ir import has_class, remap_to_species, to_ir

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/training/datasets/live"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write images/ + labels/ (default: the dataset's train split)",
    )
    p.add_argument("--source-class", type=int, default=5, help="Keep train day frames containing this class (default 5=bambi)")
    p.add_argument("--day-token", default="_day_", help="Substring marking a day frame in the slugged filename")
    p.add_argument("--prefix", default="synthir__", help="Filename prefix for generated frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Cap number generated (0 = all)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ds = args.dataset_dir.resolve()
    img_train = ds / "images" / "train"
    lbl_train = ds / "labels" / "train"
    if not lbl_train.is_dir():
        raise SystemExit(f"No train labels at {lbl_train}; run prepare_dataset first.")

    out = (args.output_dir.resolve() if args.output_dir else ds / "train_synth_stage")
    out_img = out / "images"
    out_lbl = out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    made = 0
    rows: list[dict[str, str]] = []
    for txt in sorted(lbl_train.glob("*.txt")):
        if args.day_token not in txt.name:
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        if not has_class(lines, args.source_class):
            continue
        # Locate the source image (any supported extension).
        src_img = next((img_train / f"{txt.stem}{e}" for e in IMAGE_EXTS if (img_train / f"{txt.stem}{e}").exists()), None)
        if src_img is None:
            continue

        remapped = remap_to_species(lines)
        if not remapped:
            continue

        stem = f"{args.prefix}{txt.stem}"
        with Image.open(src_img) as im:
            to_ir(im, seed=args.seed + made).save(out_img / f"{stem}.jpg", quality=90)
        (out_lbl / f"{stem}.txt").write_text("\n".join(remapped) + "\n", encoding="utf-8")
        rows.append({"source_image": str(src_img), "output_image": str(out_img / f"{stem}.jpg")})
        made += 1
        if args.limit and made >= args.limit:
            break

    with (out / "synth_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_image", "output_image"])
        w.writeheader()
        w.writerows(rows)

    print(f"Generated {made} synthetic-IR frames (source class {args.source_class}) -> {out}")


if __name__ == "__main__":
    main()
