# Annotation Workflow

Use CVAT to label bird bounding boxes. Keep the images local where possible.

## Start CVAT

CVAT changes its Docker Compose stack over time, so this project intentionally
uses the official CVAT repository instead of copying a large Compose file here.

```bash
cd annotation
git clone https://github.com/cvat-ai/cvat.git cvat
cd cvat
docker compose up -d
```

Open CVAT at `http://localhost:8080`, create a user, and create one task per
image source or capture session.

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
- `exports/`: CVAT YOLO exports, one folder per dataset version.

Keep raw image filenames descriptive enough to preserve source context:

```text
room-main_day_2026-06-08_00123.jpg
room-side_ir_2026-06-08_00042.jpg
phone_pixel8_2026-05-18_00012.jpg
```

## Export

In CVAT, export tasks as YOLO format. Unzip each export under:

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
