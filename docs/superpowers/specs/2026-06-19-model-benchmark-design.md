# Model Benchmark — Design

**Date:** 2026-06-19

## Goal

Benchmark every trained model in `data/models/` against the human-labeled
annotation boxes, write one results file (`data/models/benchmark.json`), and add
an interactive explorer to the annotation homepage that shows how each new model
**version** improved or regressed — overall and per label.

## Scope & rules

- Models are `data/models/<series>-NNN.pt` (Ultralytics YOLO). Series is the
  filename prefix:
  - **live-\*** → evaluated on `tapo/day` + `tapo/ir`
  - **archive-\*** → evaluated on `phone`
- "Benchmark" = for every **labeled box** (a box with a non-null `label`) in the
  model's categories, did the model detect it correctly?
  - A ground-truth (GT) box is a **HIT (TP)** when the model emits a detection
    with **IoU ≥ match-iou (0.5)** and the **same label**.
  - A predicted box of label L matching no GT of L → **FP**.
  - A GT box of label L matched by no prediction → **FN (miss)**.
  - `gt = tp + fn` (every GT is matched or missed).
- Per label / per category / overall (micro-averaged): **recall = tp/(tp+fn)**,
  **precision = tp/(tp+fp)**, **F1 = 2PR/(P+R)**. Null when the denominator is 0.
- Matching uses the **label string** against `model.names` (the .pt embeds its
  class names), so no roster index remapping is needed.
- A GT label outside the model's vocabulary is **excluded** from its metrics and
  counted as `naBoxes`. Images whose every labeled box is out-of-vocab are
  skipped (no inference) so precision isn't unfairly penalized.
- Only images with **≥1 in-vocab labeled box** are scored. Defaults: confidence
  0.25, match-IoU 0.5, NMS-IoU 0.7, imgsz 960, device auto (GPU if available).

## Part 1 — Script

`training/scripts/benchmark.py` (Python, ultralytics), invoked via
`scripts/benchmark.sh` / `scripts/benchmark.ps1` (mirroring `train_live.*`:
`uv sync` → `install-gpu` → `uv run --no-sync python …`) and exposed as the
`benchmark` console entry point in `pyproject.toml`.

CLI flags: `--models-dir`, `--data-root`, `--output`, `--conf`, `--iou`
(match), `--nms-iou`, `--imgsz`, `--device`, `--limit` (images/category, 0=all),
`--series`, `--models`. Writes `benchmark.json` atomically (temp + replace).

### `benchmark.json` schema (camelCase; metrics are fractions 0..1, 4 dp)

```jsonc
{
  "schemaVersion": 1,
  "generatedAt": "2026-06-19T12:34:56Z",
  "config": { "confidence": 0.25, "iou": 0.5, "nmsIou": 0.7, "imgsz": 960 },
  "series": [
    {
      "name": "live",
      "categories": ["day", "ir"],
      "models": [
        {
          "name": "live-001", "version": 1, "file": "live-001.pt",
          "images": 1234, "gtBoxes": 2345, "naBoxes": 0,
          "overall":   { "recall": .., "precision": .., "f1": .., "tp": .., "fp": .., "fn": .., "gt": .. },
          "byCategory": { "day": { ..metrics.., "images": .. }, "ir": { ..metrics.., "images": .. } },
          "labels":    { "percy": { ..metrics.., "gt": .. }, "...": {} }
        }
      ]
    },
    { "name": "archive", "categories": ["phone"], "models": [ /* … */ ] }
  ]
}
```

## Part 2 — Homepage explorer

New client component `annotation/components/BenchmarkExplorer.tsx`, rendered on
`app/page.tsx` (after the per-category progress grid). Reads `benchmark.json`
through a server-only reader `lib/benchmark.ts` (mtime-cached) via
`GET /api/benchmark` → `{ benchmark: BenchmarkFile | null }`. Client-safe types
in `lib/benchmark-types.ts`. Path constants `MODELS_DIR` / `BENCHMARK_PATH` added
to `lib/paths.ts`. Uses existing theme tokens; custom inline SVG chart (no chart
library).

### Interaction (the approved design)

- **Series tabs**: Live / Archive (only shown if present).
- **Metric toggle**: Recall / F1 / Precision (default F1).
- **Overall trend line** across model versions (x = versions, y = 0–100, labeled
  axes). **By-category** toggle (day vs ir) for multi-category series; hidden for
  single-category series (archive).
- **Scrub the line** (pointer + touch) → a crosshair snaps to the nearest version
  and the **breakdown panel updates** to that model's per-label scores (bars
  0–100, sorted by the metric) with **Δ vs the previous version** (green ▲ / red
  ▼; "new" at v1). **Click/tap to pin**; nothing pinned → **latest**.
- **Last-generated timestamp** shown.

### Edge states

- **No `benchmark.json`** (or no series/models) → minimal empty state: title +
  one line + the command (`scripts/benchmark.ps1` · `scripts/benchmark.sh`).
- **Single version** → dot + value, no trend line, hint to run a newer model.
- **Single series** → other tab hidden.

## Verification

- Web: `tsc`/`next lint`/`next build` clean.
- Script: import + a `--limit` smoke run on `live-001`/`live-002` producing a real
  `benchmark.json`, then confirm the explorer renders it.
