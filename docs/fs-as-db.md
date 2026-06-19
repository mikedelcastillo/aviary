# The filesystem as the database

The annotation tool uses the raw image tree as its database — no DB server. This
note documents the data model, how the data layer stays fast and correct, and the
operational tooling around it.

## Data model

Each image has up to three sidecars (same basename, different extension), written
next to it under `data/annotation/raw/<category>/`:

| File | Role | Authoritative? |
|---|---|---|
| `<image>.json` | Source-of-truth annotation: `{ v, boxed, boxes[] }` | **Yes** |
| `<image>.txt` | YOLO export (labeled boxes only), derived from `.json` | No (regenerable) |
| `<image>.suggest.json` | Model proposals (suggest_boxes / suggest_labels) | No (advisory) |

`v` is the schema version stamped on every write, so the format can evolve and be
migrated later. An empty `.txt` is meaningful — it marks a reviewed "negative."

## Durability

- **Atomic writes.** Both `.json` and `.txt` are written to `*.tmp-<pid>` then
  `rename`d over the target (`persistAnnotationFiles`). A crash mid-write can't
  leave a torn file — the rename either fully replaces the target or doesn't.
- **Per-file locking.** `withFileLock` serializes concurrent writers to the same
  sidecar (autosave + grid box-ops) so updates can't be lost.
- **Crash cleanup.** Leftover `*.tmp-<pid>` files are swept once on first manifest
  build (`ensureTempCleanup`).
- **Corruption is loud.** `readAnnotation` distinguishes "no sidecar yet" (ENOENT —
  seed from `.txt`) from "sidecar exists but won't parse" (throws), so the editor
  never silently overwrites a recoverable-but-corrupt file. Read-only scans log a
  corrupt file once and skip it.

## Read performance

The home page fans out several endpoints at once (`progress`, `queue`,
`box-queue`, `entry`, `label-stats`), each scanning the whole manifest.

- **Bounded I/O** (`lib/fs-limit.ts`). A global semaphore caps concurrent
  reads/stats (64), so the fan-out can't exhaust file handles (EMFILE).
- **Unified state cache** (`lib/state-cache.ts`). Each `<image>.json` is read and
  parsed at most once per change and summarized into a compact `JsonState`
  (`boxed`, `labeled`, `hasUnlabeled`, `boxLabels`). Every view derives from this
  one cache, so warm scans skip `readFile` + `JSON.parse` entirely.
- **mtime validation.** Each lookup still `stat`s the file (cheap, bounded), so an
  edit by an external script is detected via mtime and re-read. The filesystem
  stays the source of truth.

## Freshness with external writers

`immich` imports and the training/suggest scripts write files behind the app's
back. The manifest cache detects added/removed files automatically by hashing the
category directories' mtimes (`categoryDirSignature`) — no manual invalidation
needed. `GET /api/generation` returns a monotonic token that bumps on every
manifest rebuild, so a client can poll cheaply and refetch only on change.

## Operations

- **Integrity check:** `node scripts/doctor-fsdb.mjs` reports corrupt JSON,
  orphaned sidecars (no matching image), leftover `.tmp` files, empty `.txt`, and
  un-annotated images. Exits non-zero if anything needs attention.
- **Backups.** Sidecars are tiny and text — back them up independently of the
  heavy images. From the repo root:
  ```sh
  # snapshot just the annotations (json + txt), excluding images
  tar czf annotations-$(date +%Y%m%d).tgz \
    $(find data/annotation/raw -name '*.json' -o -name '*.txt')
  ```
  Restoring is just untarring over the tree. Because the FS is the DB, ordinary
  file tooling (rsync, git, zip, cloud sync) is your backup/restore layer.

## When to add a real index

Everything above keeps O(n) scans fast enough at ~12k images. If queries grow
(cross-label filtering/sorting at 100k+), add SQLite as a *derived index only* —
rebuilt from the sidecars, never the source of truth — to keep the
inspect/backup and external-tooling benefits while gaining indexed queries.
