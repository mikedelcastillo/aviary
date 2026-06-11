# Annotation Workflow

Label bird bounding boxes against the shared roster, then save YOLO-format
labels for training. Keep the images local where possible.

## Annotation Tool

> **TODO: build a custom bird tagger.** We tried CVAT and dropped it — a
> 17-container stack is far too heavy for one annotator, and it has no notion of
> "which images are already labeled," so re-dumping a folder re-imports work
> you've already done. A small, purpose-built tagger will replace it: read
> images straight from `raw/`, skip the ones that already have a label, draw one
> box per bird, pick a label from `roster.yaml`, and write a YOLO `.txt` sidecar
> directly — no separate export/unzip step.

Until the tagger exists, label with whatever tool you like. The only contract is
the output: YOLO-format `.txt` files under `exports/` (see **Export**). The
label **order in `roster.yaml` is the integer class index**, so the labels must
be emitted in that order.

## Label Schema

Use the labels in `roster.yaml` — the single roster shared by both the live and
archive models. Label **every** bird against this one schema; the per-model
class split happens later at dataset-prep time (`prepare-dataset --model …`), so
you never think about models while labeling.

Each visible bird gets exactly one bounding box. On **color** frames label the
individual name; on **IR/night** frames label the species (`cockatiel`,
`lovebird`, `budgie`) only. Use `unknown_bird` when identity is not clear. See
the `rules:` block in `roster.yaml` and `../README.md` for the full rationale.

Do not draw boxes around reflections, toys, shadows, or bird-shaped objects.

## Folder Use

- `raw/phone_photos/`: phone images from your library.
- `raw/immich_birds/`: images from Immich `Birds` albums, pulled with the standalone
  [immich-auto-albums](https://github.com/mikedelcastillo/immich-auto-albums) tool.
- `raw/camera_frames/day/`: extracted visible-light camera frames.
- `raw/camera_frames/ir/`: extracted infrared/night-mode frames.
- `exports/`: YOLO-format label exports, one folder per dataset version.

Keep raw image filenames descriptive enough to preserve source context:

```text
room-main_day_2026-06-08_00123.jpg
room-side_ir_2026-06-08_00042.jpg
phone_pixel8_2026-05-18_00012.jpg
```

## Export

Each labeled batch becomes a YOLO-format export — images plus their `.txt`
label files — placed under a versioned folder:

```text
models/annotation/exports/v001/
models/annotation/exports/v002/
```

Then normalize the export for training:

```bash
python models/training/scripts/prepare_dataset.py \
  --source models/annotation/exports/v001 \
  --output models/training/datasets/v001
```
