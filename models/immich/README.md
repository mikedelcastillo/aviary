# Immich Bird Album Import

This folder pulls candidate bird photos from your local Immich server before
CVAT labeling.

It has two entrypoints:

- `generate_albums.py` (`uv run generate-albums`): scan each configured Immich
  account, detect photos that likely contain birds, and add them to that
  account's `Birds` album.
- `download_immich_birds.py` (`uv run download-birds`): download all images in
  each account's `Birds` album into
  `models/annotation/raw/immich_birds/<account>/`.

## Setup

Copy the environment template:

```bash
cp .env.example .env
```

Edit `.env`:

```text
IMMICH_BASE_URL=http://192.168.1.168:2283/api
IMMICH_ACCOUNT_1_API_KEY=replace-me
IMMICH_ACCOUNT_2_API_KEY=replace-me
```

Edit `models/immich/config/accounts.yaml` so every Immich account has a slug and
an API key env var name. The config is safe to commit because the actual API key
values stay in `.env`. Create API keys in Immich under User Settings -> API Keys.

Install dependencies from the repo root (see the top-level README for installing uv):

```bash
uv sync                  # installs everything (with a default CPU/CUDA torch)
./scripts/install-gpu.sh # one time: installs the right GPU torch for THIS machine
```

`uv sync` can't pick the GPU build itself — the GTX 1060 and RTX 5060 both look like
`linux/x86_64` to uv yet need incompatible wheels (cu118/Pascal vs cu128/Blackwell), so
`install-gpu.sh` detects the GPU and installs the matching torch into the venv. On Windows
use `.\scripts\install-gpu.ps1` (AMD Radeon → native ROCm wheels from `repo.radeon.com`,
which need **Python 3.12** + **AMD driver 26.2.2**).

GPU notes:

- **NVIDIA Pascal (GTX 1060)** runs **fp32** — fp16 is ~1/64 speed on Pascal and is
  auto-disabled. **Blackwell (RTX 5060)** needs the `cu128` build.
- fp16 selection is automatic; override with `IMMICH_BIRD_HALF=1` / `0` if needed.
- On Apple Silicon, omit `install-gpu` — the default torch uses `mps`.

## Generate Birds Albums

Run with `--no-sync` so uv keeps the GPU torch from `install-gpu` instead of reverting it to
the default build. Start with a dry run:

```bash
uv run --no-sync generate-albums --dry-run --limit 25
```

Then run for real:

```bash
uv run --no-sync generate-albums
```

(Tip: `export UV_NO_SYNC=1` once in your shell profile and you can drop the flag —
`uv run generate-albums` then works as-is.)

Defaults:

- model: `yolo11x.pt`
- class filter: COCO `bird`
- threshold: `0.30`
- device: `auto` (`cuda:0`, then `mps`, then `cpu`)
- inference batch: `64`, auto-halved on CUDA OOM (the 6 GB GTX 1060 self-tunes to 32)
- album name: `Birds`

## Download Bird Album Images

```bash
uv run download-birds
```

Files are saved under:

```text
models/annotation/raw/immich_birds/<account>/
```

The scripts write resumable state and manifests under `models/immich/state/` and
`models/immich/manifests/`.

## Notes

This is a high-recall prefilter. Expect false positives. The authoritative
labels still come from CVAT.

Immich endpoints used:

- `POST /search/metadata`
- `GET /users/me`
- `GET /albums`
- `POST /albums`
- `PUT /albums/{id}/assets`
- `GET /assets/{id}/thumbnail`
- `GET /assets/{id}/original`
