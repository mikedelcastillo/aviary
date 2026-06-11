# Aviary Annotation Tool

A local, single-user web tool for drawing and labeling bird bounding boxes. It
reads the raw images under `data/annotation/raw/` and writes YOLO-format labels
directly back next to them — the **filesystem is the database**, there is no
server DB and no authentication.

## Run

```bash
./scripts/annotation.sh
```

Then open **http://0.0.0.0:5000**. First run installs npm dependencies; Node 18+
is required (the repo is otherwise Python/uv — this is the only JS/TS app).

## How it works

Each raw image moves through two phases:

1. **Box** (`/box/[n]`) — draw/vet tight bounding boxes. Tapo images arrive
   pre-boxed by the detector; those seed boxes are editable but an image only
   counts as *boxed* once you review and advance past it. Click-drag to add a
   box, hover a box's outline to reveal an **✕** to delete it, `Cmd/Ctrl+Z` /
   `Cmd/Ctrl+Y` to undo/redo, arrow keys to navigate, `f` to fit.
2. **Label** (`/label/[n]`) — the image spotlights one box at a time; pick a
   label from the bottom pills (each has a keyboard shortcut). Labeling a box
   auto-advances to the next box that needs one.

Categories: `tapo/day`, `tapo/ir`, `phone`. Label pills come from
[`models/roster.yaml`](../models/roster.yaml) and depend on the category:

- **Tapo · Day** → living individual bird names (live model, `kind: individual`).
- **Tapo · IR** → species only (`cockatiel`, `lovebird`, `budgie`).
- **Phone** → every individual name (archive model).

`unknown_bird` is always available. Pill order follows roster file order, which
is also the integer YOLO class index.

## Storage

Per image, next to the `.jpg`:

- `<image>.json` — source of truth: `{ boxed, boxes:[{id,cx,cy,w,h,label}] }`
  (normalized YOLO center+size; `label` is a roster name or `null`).
- `<image>.txt` — YOLO export, rewritten on every save: only labeled boxes, at
  their **global roster class index**. This doubles as the training export, so
  `prepare-dataset` can consume the labels with no separate export step.

Everything autosaves (debounced); the browser tab title shows `Saving…` → `Saved`.

## Config

The launcher sets these (absolute) env vars; override if your layout differs:

- `AVIARY_DATA_ROOT` — raw image root (default `data/annotation/raw`).
- `AVIARY_ROSTER` — roster YAML (default `models/roster.yaml`).

## Stack

Next.js 15 (App Router) · TypeScript · Tailwind v4 · Geist · Zustand. The canvas
is an SVG overlay over an `<img>` inside one CSS-transformed stage (crisp vector
boxes, native pan/zoom). See `lib/` for the data contract and `components/` for
the canvas.
