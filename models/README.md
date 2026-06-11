# Bird Models — Strategy & Labeling Rules

This is the authoritative guide for how we build the bird detector/identifier.
It documents *why* the dataset and class scheme look the way they do. The
training mechanics live in [`training/README.md`](training/README.md) (dataset
prep + training); labeling is done in the `/annotation` web tool (run
`./scripts/annotation.sh`).

Deployment target: **Tapo CCTV**. Everything below optimizes for that domain.

## Scope: two models, one roster

We train **two** models from a **single label roster**
([`roster.yaml`](roster.yaml)):

| | **Live model** | **Archive model** |
|---|---|---|
| Purpose | Real-time ID on the CCTV feed | Catalog the photo library |
| Classes | 6 living birds + 3 IR species + `unknown_bird` | All individuals (living + deceased), names only |
| Domains | RGB + IR | RGB only — no IR, no species classes |
| Roster tag | `models: [live]` (or `[live, archive]`) | `models: [archive]` |
| Grows? | Fixed roster | Fixed historical set |

Deceased birds belong **only** to the archive model — they never appear on the
live feed, so naming them there only invites confusing a living bird for a dead
one. You label every bird once against the roster; `prepare-dataset --model
live|archive` filters and remaps to each model's class set. Most of this doc
concerns the **live** model; the archive model is just "all individuals, RGB,
flat classes."

## The flock

Six birds. They split cleanly into a large crested group (cockatiels) and a
small group (lovebirds + budgie), which is why **size/silhouette alone**
separates those groups even in infrared. The hard part is telling birds *within*
a group apart once color is gone.

| Bird (rename to pet name) | Description | Size / shape | IR (grayscale) signature |
|---|---|---|---|
| `cockatiel_whiteface` | All white with some grey | Large, crested, long tail | Pale body, mid-grey patches |
| `cockatiel_lutino` | Red cheek, yellow crest, light yellow body | Large, crested, long tail | Pale body + dark cheek spot |
| `lovebird_orange_masked` | Orange head, green wings | Small, stocky, stub tail | Mid head/body — low contrast |
| `lovebird_black_masked` | Black head, blue wings | Small, stocky, stub tail | **Dark head** — distinctive |
| `lovebird_green` | All green | Small, stocky, stub tail | Uniform mid-grey |
| `budgie_yellow` | All yellow | Small, slender, longer tail | Bright/pale, slender silhouette |

### Hard IR pairs (over-collect these)

When color is gone, two pairs become hard to tell apart:

1. **`cockatiel_whiteface` vs `cockatiel_lutino`** — both are pale birds in IR.
2. **`lovebird_green` vs `lovebird_orange_masked`** — same size/shape; only a
   subtle head/body tonal contrast separates them.

Everything else (cockatiel-vs-small-bird, the black-masked lovebird, budgie-vs-
lovebird) stays reliable in IR via size/shape/tone.

## Model & class scheme

**One model, all permutations mixed (day/IR × caged/free × CCTV/phone).** Do not
build a model per permutation — YOLO learns shared features across domains and
one model is far less to maintain.

The class scheme encodes "name by day, species by night" directly into the
labels — **9 classes**:

- **6 named classes** — applied **only to color (RGB) frames**.
- **3 species classes** (`cockatiel`, `lovebird`, `budgie`) — applied **only to
  IR frames**. We do not need individual names at night.
- (`unknown_bird` for visible-but-unidentifiable birds, either domain.)

IR frames are grayscale — a massive, unmistakable signal — so the model trivially
learns "color cockatiel → name it, grayscale cockatiel → just `cockatiel`." This
gives named daytime ID and species-only night tagging from a single model, with
**no post-hoc confidence threshold or routing logic**. The labeling scheme *is*
the abstain-at-night rule.

The canonical label list lives in
[`roster.yaml`](roster.yaml). The live model's classes are
the roster entries tagged `live`, renumbered to a contiguous range by
`prepare-dataset --model live`.

## Labeling rules

1. **Label by identity only — location is never a class.** A bird is the same
   bird in or out of its cage. Do **not** create in-cage/out-of-cage classes.
   Caged and free are *data diversity*, not labels — include both.
2. **RGB frame → named class. IR frame → species class.** Keep the regimes
   clean: never put a species label on a color frame, never put a name on an IR
   frame.
3. **All IR birds get species labels — no exceptions.** The black-masked
   lovebird is identifiable in IR, but since we don't need night names, label it
   `lovebird` like the rest. Mixed rules teach the model a confusing boundary.
4. Draw one tight box per visible bird; include partially occluded / behind-bars
   birds if meaningful body area shows.
5. Use `unknown_bird` when a bird is visible but identity is unclear.
6. Do not box toys, shadows, reflections, cage hardware, or empty perches.

## Data sources & splits

- **CCTV is the primary domain — keep it the majority of the training set.**
  `raw/tapo/day/` and `raw/tapo/ir/`.
- **Phone photos are a supplement, not the bulk** (`raw/phone/`).
  They are RGB-only, so they help the daytime case only —
  they do nothing for IR. Their highest value is crisp close-ups of the **hard
  pairs** (cheek spot, mask), which teach discriminative features a blurry CCTV
  crop cannot. Different domain (close, eye-level, sharp) — if they dominate, the
  model optimizes for a domain we never deploy in.
- **Validation/test set must be CCTV-only and include IR frames.** Evaluating on
  easy phone close-ups produces flattering numbers that lie about real-world
  performance.

## Sample targets

- **~150–300 images per bird**, spread across all permutations (caged/free,
  day/IR, varied pose/angle/distance/lighting).
- **Diversity beats count.** 200 varied images beat 1000 near-duplicate frames.
  CCTV floods you with near-duplicates — dedupe / sample (perceptual hash or 1
  frame every N seconds + manual cull).
- **Every bird needs IR samples.** A bird that only appears in daytime RGB is
  unrecognizable at night, regardless of total count.
- **Over-collect IR for the two hard pairs** above — the tonal cues only survive
  certain angles, so you need many.

## Runtime behavior

- **Names by day, species at night** — falls out of the class scheme; no
  thresholding needed.
- **Dusk transition:** when the Tapo flips IR on there's a brief window of
  borderline-grayscale frames where output may flip between a name and a species.
  Narrow and harmless for alerting; if it ever matters, route explicitly by
  detecting grayscale-ness per frame and only trust named outputs on color
  frames.
- **Bounding-box size is a reliable sanity check** — cockatiel vs small-bird is
  unambiguous by size even when identity confidence is low.

## Why this shape (rationale)

- **Single-stage 9-class YOLO, not a two-stage detect→re-ID pipeline.** Two-stage
  (embeddings / metric learning) earns its complexity when birds look alike or
  the flock grows. Ours are well-separated and fixed at six. Keep it simple.
  Revisit two-stage only if the hard IR pairs stay bad after more targeted data.
- **One model, not per-permutation models.** Less maintenance, shared features,
  one training run. Track day and IR metrics *separately* during evaluation; if
  IR is weak, add IR frames before touching architecture.
