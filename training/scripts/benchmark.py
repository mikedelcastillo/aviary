#!/usr/bin/env python3
"""Benchmark every model in data/models against the labeled annotation boxes.

For each model file ``<series>-NNN.pt`` in the models directory:

* ``live-*``    -> evaluated on tapo/day + tapo/ir
* ``archive-*`` -> evaluated on phone

"Benchmark" = for every human-labeled box (a box with a non-null ``label``) in the
model's categories, did the model detect it? A ground-truth box is a HIT when the
model emits a detection with IoU >= --iou AND the same label. Per label we
accumulate hits (TP), false positives (FP -- a predicted box of that label matching
no GT) and misses (FN -- a GT box matched by no prediction), then derive recall,
precision and F1. Out-of-vocabulary GT labels (a label the model cannot predict)
are excluded from its metrics and counted separately as ``naBoxes``.

Matching is on the *label string* against ``model.names`` (the .pt embeds its class
names), so no roster index remapping is needed.

Results -- per label, per category and overall, for every model -- are written to
data/models/benchmark.json, grouped into the ``live`` and ``archive`` series so the
annotation homepage can chart how each new model version improved or regressed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Category id -> directory under the data root. Mirrors annotation/lib/types.ts
# CATEGORIES (day, ir, phone) so the script scores exactly the labeled tree the
# annotation tool writes.
CATEGORY_DIRS: dict[str, Path] = {
    "day": Path("tapo") / "day",
    "ir": Path("tapo") / "ir",
    "phone": Path("phone"),
}

# Which categories each model series is benchmarked against (per the roster rules
# in training/roster.yaml): live = real-time CCTV (day + ir), archive = phone.
SERIES_CATEGORIES: dict[str, list[str]] = {
    "live": ["day", "ir"],
    "archive": ["phone"],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# data/models/<series>-NNN.pt -> (series, version). Matches train_live/​train_archive
# export naming (zero-padded sequence).
MODEL_RE = re.compile(r"^([A-Za-z]+)-(\d+)$")

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)

# The benchmark scores at the SAME confidence the live server uses, so its numbers
# reflect what the deployed detector actually does. Single source of truth: the
# server's ModelConfig (server/lib/config.py). Falls back to the historical default
# only if the server package isn't importable (e.g. a stripped training env).
try:
    from lib.config import ModelConfig as _ServerModelConfig

    _DEFAULT_CONF = _ServerModelConfig.confidence
except Exception:
    _DEFAULT_CONF = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"), help="Directory of <series>-NNN.pt models")
    parser.add_argument("--data-root", type=Path, default=Path("data/annotation/raw"), help="Labeled annotation root")
    parser.add_argument("--output", type=Path, default=Path("data/models/benchmark.json"), help="Results JSON path")
    parser.add_argument(
        "--conf",
        type=float,
        default=_DEFAULT_CONF,
        help="Detection confidence threshold (default mirrors the server: "
        "server/lib/config.py ModelConfig.confidence)",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching a detection to a GT box")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU passed to the model's predict()")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    parser.add_argument("--limit", type=int, default=0, help="Max labeled images to score per category (0 = all)")
    parser.add_argument("--series", action="append", choices=sorted(SERIES_CATEGORIES), help="Only this series (repeatable)")
    parser.add_argument("--models", action="append", help="Only these model filenames/stems (repeatable)")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Score only this dataset split (default all). 'test' = the held-out set "
        "from the manifest, so the score isn't measured on training images.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/training/datasets"),
        help="Where <series>/manifest.csv lives (used only when --split is not 'all')",
    )
    return parser.parse_args(argv)


def load_split_sources(dataset_dir: Path, series: str, split: str) -> set[Path] | None:
    """Resolved source-image paths in a dataset split, or ``None`` if no manifest.

    Lets the benchmark score ONLY a held-out split (e.g. ``test``) instead of the
    whole labeled tree (which is mostly training data and flatters the score). The
    manifest is written by ``prepare_dataset.py`` at dataset-build time.
    """
    manifest = dataset_dir / series / "manifest.csv"
    if not manifest.exists():
        return None
    allowed: set[Path] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split and row.get("source_image"):
                allowed.add(Path(row["source_image"]).resolve())
    return allowed


# --- Geometry ---------------------------------------------------------------

def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# --- Ground truth -----------------------------------------------------------

def read_gt(image: Path) -> list[tuple[str, Box]]:
    """Labeled GT boxes for an image from its `.json` sidecar (xyxy normalized).

    Only boxes with a non-null label count; an unboxed/empty/missing/corrupt
    sidecar yields no GT (the image is then skipped by the caller).
    """
    sidecar = image.with_suffix(".json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return []
    out: list[tuple[str, Box]] = []
    for raw in data.get("boxes", []) or []:
        label = raw.get("label")
        if not label:
            continue
        try:
            cx, cy, w, h = float(raw["cx"]), float(raw["cy"]), float(raw["w"]), float(raw["h"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((str(label), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def list_images(category_dir: Path) -> list[Path]:
    if not category_dir.exists():
        return []
    images = [
        p
        for p in category_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and "__MACOSX" not in p.parts
    ]
    images.sort()
    return images


# --- Counts / metrics -------------------------------------------------------

@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def _round(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def metrics(c: Counts) -> dict:
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) > 0 else None
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "recall": _round(recall),
        "precision": _round(precision),
        "f1": _round(f1),
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "gt": c.tp + c.fn,  # every GT box is either matched (tp) or missed (fn)
    }


def match_image(gts: list[tuple[str, Box]], preds: list[tuple[str, float, Box]], match_iou: float) -> dict[str, Counts]:
    """Greedy per-label matching (predictions by descending confidence)."""
    res: dict[str, Counts] = defaultdict(Counts)
    gts_by_label: dict[str, list[Box]] = defaultdict(list)
    for label, box in gts:
        gts_by_label[label].append(box)
    matched: dict[str, list[bool]] = {label: [False] * len(boxes) for label, boxes in gts_by_label.items()}

    for label, _conf, pbox in sorted(preds, key=lambda p: p[1], reverse=True):
        candidates = gts_by_label.get(label, [])
        best_i, best_iou = -1, match_iou
        for i, gbox in enumerate(candidates):
            if matched[label][i]:
                continue
            value = iou(pbox, gbox)
            if value >= best_iou:
                best_iou, best_i = value, i
        if best_i >= 0:
            matched[label][best_i] = True
            res[label].tp += 1
        else:
            res[label].fp += 1

    for label, flags in matched.items():
        res[label].fn += sum(1 for f in flags if not f)
    return res


# --- Inference --------------------------------------------------------------

def predict_boxes(model, image: Path, conf: float, nms_iou: float, imgsz: int, device: str) -> list[tuple[str, float, Box]]:
    args = {"source": str(image), "conf": conf, "iou": nms_iou, "imgsz": imgsz, "verbose": False}
    if device != "auto":
        args["device"] = device
    results = model.predict(**args)
    if not results:
        return []
    names = model.names
    out: list[tuple[str, float, Box]] = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxyn[0].tolist())
        if isinstance(names, dict):
            label = str(names.get(cls_id, cls_id))
        else:
            label = str(names[cls_id]) if cls_id < len(names) else str(cls_id)
        out.append((label, confidence, (x1, y1, x2, y2)))
    return out


@dataclass
class GatheredImage:
    image: Path
    gts: list[tuple[str, Box]]  # in-vocab GT only


@dataclass
class ModelResult:
    name: str
    version: int
    file: str
    images: int
    na_boxes: int
    cat_counts: dict[tuple[str, str], Counts]  # (category, label) -> counts
    cat_images: dict[str, int] = field(default_factory=dict)


def evaluate_model(model_path: Path, series: str, categories: list[str], args) -> ModelResult:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    vocab = set(model.names.values()) if isinstance(model.names, dict) else set(model.names)

    # Gather first (cheap sidecar reads) so we have an accurate progress total and
    # can apply --limit deterministically.
    allowed: set[Path] | None = None
    if args.split != "all":
        allowed = load_split_sources(args.dataset_dir, series, args.split)
        if allowed is None:
            print(
                f"  no manifest at {args.dataset_dir / series}/manifest.csv; scoring ALL "
                f"{series} images (run prepare-dataset to get a held-out {args.split} split)"
            )

    gathered: dict[str, list[GatheredImage]] = {}
    na_total = 0
    for cat in categories:
        cat_dir = (args.data_root / CATEGORY_DIRS[cat]).resolve()
        items: list[GatheredImage] = []
        for image in list_images(cat_dir):
            if allowed is not None and image.resolve() not in allowed:
                continue
            gts = read_gt(image)
            if not gts:
                continue
            in_vocab = [(label, box) for label, box in gts if label in vocab]
            na_total += len(gts) - len(in_vocab)
            if not in_vocab:
                continue  # all labels out-of-vocab: skip inference, don't penalize precision
            items.append(GatheredImage(image=image, gts=in_vocab))
            if args.limit and len(items) >= args.limit:
                break
        gathered[cat] = items

    cat_counts: dict[tuple[str, str], Counts] = defaultdict(Counts)
    cat_images: dict[str, int] = {cat: len(items) for cat, items in gathered.items()}
    total = sum(cat_images.values())

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

    stem = model_path.stem
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("imgs"),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task(stem, total=max(1, total))
        for cat, items in gathered.items():
            for item in items:
                preds = predict_boxes(model, item.image, args.conf, args.nms_iou, args.imgsz, args.device)
                per_label = match_image(item.gts, preds, args.iou)
                for label, counts in per_label.items():
                    cat_counts[(cat, label)].add(counts)
                progress.advance(task)

    return ModelResult(
        name=stem,
        version=int(MODEL_RE.match(stem).group(2)),
        file=model_path.name,
        images=total,
        na_boxes=na_total,
        cat_counts=cat_counts,
        cat_images=cat_images,
    )


def result_to_json(res: ModelResult, categories: list[str]) -> dict:
    labels = sorted({label for (_cat, label) in res.cat_counts})

    def sum_over(keys: list[tuple[str, str]]) -> Counts:
        total = Counts()
        for key in keys:
            total.add(res.cat_counts[key])
        return total

    labels_out = {}
    for label in labels:
        keys = [(cat, label) for cat in categories if (cat, label) in res.cat_counts]
        labels_out[label] = metrics(sum_over(keys))

    by_category = {}
    for cat in categories:
        keys = [(cat, label) for label in labels if (cat, label) in res.cat_counts]
        entry = metrics(sum_over(keys))
        entry["images"] = res.cat_images.get(cat, 0)
        by_category[cat] = entry

    overall = metrics(sum_over(list(res.cat_counts.keys())))

    return {
        "name": res.name,
        "version": res.version,
        "file": res.file,
        "images": res.images,
        "gtBoxes": overall["gt"],
        "naBoxes": res.na_boxes,
        "overall": overall,
        "byCategory": by_category,
        "labels": labels_out,
    }


def discover_models(models_dir: Path, only_series, only_models) -> list[tuple[str, int, Path]]:
    """Return (series, version, path) for each benchmarkable model, sorted."""
    wanted = None
    if only_models:
        wanted = {Path(m).stem for m in only_models}
    found: list[tuple[str, int, Path]] = []
    for path in sorted(models_dir.glob("*.pt")):
        match = MODEL_RE.match(path.stem)
        if not match:
            continue
        series, version = match.group(1).lower(), int(match.group(2))
        if series not in SERIES_CATEGORIES:
            print(f"  skip {path.name}: unknown series '{series}' (expected one of {sorted(SERIES_CATEGORIES)})")
            continue
        if only_series and series not in only_series:
            continue
        if wanted is not None and path.stem not in wanted:
            continue
        found.append((series, version, path))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    models = discover_models(args.models_dir, args.series, args.models)
    if not models:
        raise SystemExit(f"No benchmarkable models in {args.models_dir} (expected <series>-NNN.pt).")

    print(
        f"Benchmarking {len(models)} model(s) — split={args.split} conf={args.conf} "
        f"iou={args.iou} imgsz={args.imgsz} device={args.device}"
    )

    series_models: dict[str, list[dict]] = defaultdict(list)
    for series, _version, path in models:
        categories = SERIES_CATEGORIES[series]
        print(f"\n{path.name}  ->  {series} ({', '.join(categories)})")
        result = evaluate_model(path, series, categories, args)
        model_json = result_to_json(result, categories)
        ov = model_json["overall"]
        print(
            f"  overall recall={_fmt(ov['recall'])} precision={_fmt(ov['precision'])} f1={_fmt(ov['f1'])}"
            f"  (tp={ov['tp']} fp={ov['fp']} fn={ov['fn']}, {result.images} imgs"
            + (f", {result.na_boxes} out-of-vocab" if result.na_boxes else "")
            + ")"
        )
        series_models[series].append(model_json)

    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"confidence": args.conf, "iou": args.iou, "nmsIou": args.nms_iou, "imgsz": args.imgsz, "split": args.split},
        "series": [
            {"name": name, "categories": SERIES_CATEGORIES[name], "models": series_models[name]}
            for name in SERIES_CATEGORIES
            if name in series_models
        ],
    }
    write_atomic(args.output, payload)
    print(f"\nWrote {args.output}")


def _fmt(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "—"


if __name__ == "__main__":
    main()
