# Immich Bird Album Import

This folder pulls candidate bird photos from your local Immich server before
CVAT labeling.

It has two entrypoints:

- `generate_albums.py`: scan each configured Immich account, detect photos that
  likely contain birds, and add them to that account's `Birds` album.
- `download_immich_birds.py`: download all images in each account's `Birds`
  album into `models/annotation/raw/immich_birds/<account>/`.

## Setup

Copy the account config and environment template:

```bash
cp models/immich/config/accounts.example.yaml models/immich/config/accounts.yaml
cp .env.example .env
```

Edit `.env`:

```text
IMMICH_BASE_URL=http://192.168.1.168:2283/api
IMMICH_MIKE_API_KEY=replace-me
IMMICH_OTHER_API_KEY=replace-me
```

Edit `models/immich/config/accounts.yaml` so every Immich account has a slug and an
API key env var name. Create API keys in Immich under User Settings -> API Keys.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r models/immich/requirements.txt
```

For CUDA, install the PyTorch build that matches your local NVIDIA stack before
installing or running Ultralytics. On Apple Silicon, the script auto-selects
`mps` when available.

## Generate Birds Albums

Start with a dry run:

```bash
python models/immich/generate_albums.py --dry-run --limit 25
```

Then run for real:

```bash
python models/immich/generate_albums.py
```

Defaults:

- model: `yolo11x.pt`
- class filter: COCO `bird`
- threshold: `0.30`
- device: `auto` (`cuda:0`, then `mps`, then `cpu`)
- album name: `Birds`

## Download Bird Album Images

```bash
python models/immich/download_immich_birds.py
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
