"""VLM eval tasks: observation decoration, scene description, camera naming.

Inputs replay REAL production data so the candidate sees exactly what the live
pipeline feeds it:

* decoration — observations sampled from ``data/server/memories/*.jsonl``
  (stored YOLO detections redrawn with :func:`annotate_for_vlm`, then
  :func:`analyze_frame` with ``max_dim=None`` — byte-identical to the backfill
  path). The stored qwen2.5vl:7b decorations serve as a soft reference; an
  optional hand-curated golden manifest overrides them where present.
* scene — collect sidecar frames (single detection, known label/bbox) through
  :func:`describe_scene` with the same detection-context grounding as /find.
* camera naming — distinct camera views from the memories images through
  :func:`name_camera_view`.
"""

from __future__ import annotations

import glob
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from lib.activity import _BoxDetection, _parse_sidecar
from lib.ai.evals import checks
from lib.ai.vlm import (
    analyze_frame,
    build_detection_context,
    describe_scene,
    name_camera_view,
)
from lib.memory_build import annotate_for_vlm
from lib.roster import load_sexes, pronoun_map


# Deterministic sampling: every model must be graded on the SAME frames or the
# comparison is noise. Seed fixed; samples ordered by (day, time).
SAMPLE_SEED = 20260802

GOLDEN_MANIFEST = Path("data/server/model_evals/golden_analyze.json")

# Soft-agreement classes: activities that are plausibly interchangeable for the
# same pose (a calm perched bird is fairly called resting OR alert). Exact
# match scores 1, same-class 1, different-class 0.
ACTIVITY_CLASSES = {
    "resting": "calm", "sleeping": "calm", "alert": "calm",
    "playing": "active", "exploring": "active", "climbing": "active",
    "interacting": "active", "flying": "active",
    "feeding": "food", "drinking": "food",
    "preening": "groom", "bathing": "groom",
    "vocalizing": "vocal",
    "hidden": "unseen", "unknown": "unseen",
}


@dataclass
class ObsSample:
    photo: Path
    camera: str
    labels: list[str]
    detections: list[_BoxDetection]
    # Stored incumbent decoration per label (silver reference), may be empty.
    silver: dict[str, str] = field(default_factory=dict)
    golden: dict[str, list[str]] = field(default_factory=dict)


def _iter_journal_observations(memories_dir: Path):
    for day_file in sorted(memories_dir.glob("2026-*.jsonl")):
        for line in day_file.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            for obs in entry.get("observations") or []:
                yield obs


def sample_observations(
    memories_dir: Path = Path("data/server/memories"), limit: int = 24
) -> list[ObsSample]:
    """A deterministic, diverse sample of decorated journal observations.

    Diversity knobs: spread across cameras and label-sets, prefer observations
    whose stored decoration names a NON-'calm' activity (feeding/bathing/
    playing carry the real signal — an all-'alert' sample can't distinguish
    models), and require the photo to still exist on disk.
    """
    candidates: list[ObsSample] = []
    for obs in _iter_journal_observations(memories_dir):
        photo = Path(str(obs.get("photo", "")))
        detections = obs.get("detections") or []
        if not photo.exists() or not detections or len(detections) > 4:
            continue
        boxes, silver = [], {}
        ok = True
        for det in detections:
            label = str(det.get("label", "")).strip().lower()
            bbox = det.get("bbox") or []
            if not label or len(bbox) != 4:
                ok = False
                break
            boxes.append(_BoxDetection(label, tuple(int(v) for v in bbox)))
            activity = str(det.get("activity", "")).strip().lower()
            if activity:
                silver[label] = activity
        if not ok or not silver:
            continue  # undecorated observations carry no reference signal
        candidates.append(
            ObsSample(
                photo=photo,
                camera=str(obs.get("camera", "")),
                labels=sorted({b.label for b in boxes}),
                detections=boxes,
                silver=silver,
            )
        )

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(candidates)
    # Greedy diverse pick: prefer unseen (camera, label-set) pairs and
    # non-'calm' silver activities; fill the remainder arbitrarily.
    chosen: list[ObsSample] = []
    seen_keys: set[tuple] = set()
    interesting = [
        s for s in candidates
        if any(ACTIVITY_CLASSES.get(a) not in ("calm", "unseen") for a in s.silver.values())
    ]
    for pool in (interesting, candidates):
        for sample in pool:
            if len(chosen) >= limit:
                break
            key = (sample.camera, tuple(sample.labels))
            if key in seen_keys and len(seen_keys) < limit:
                continue
            if sample in chosen:
                continue
            seen_keys.add(key)
            chosen.append(sample)
    _apply_golden(chosen)
    return chosen[:limit]


def _apply_golden(samples: list[ObsSample]) -> None:
    """Attach hand-curated golden activity sets (keyed by photo path)."""
    if not GOLDEN_MANIFEST.exists():
        return
    try:
        manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    except ValueError:
        return
    by_photo = {str(item.get("photo")): item for item in manifest.get("frames", [])}
    for sample in samples:
        item = by_photo.get(str(sample.photo))
        if item:
            sample.golden = {
                str(k).lower(): [str(v).lower() for v in vs]
                for k, vs in (item.get("allowed") or {}).items()
            }


def golden_frames() -> list[ObsSample]:
    """Samples for every frame in the golden manifest (curated set only)."""
    if not GOLDEN_MANIFEST.exists():
        return []
    manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    samples: list[ObsSample] = []
    for item in manifest.get("frames", []):
        photo = Path(str(item.get("photo", "")))
        if not photo.exists():
            continue
        boxes = [
            _BoxDetection(str(d["label"]).lower(), tuple(int(v) for v in d["bbox"]))
            for d in item.get("detections", [])
        ]
        if not boxes:
            continue
        samples.append(
            ObsSample(
                photo=photo,
                camera=str(item.get("camera", "")),
                labels=sorted({b.label for b in boxes}),
                detections=boxes,
                silver={},
                golden={
                    str(k).lower(): [str(v).lower() for v in vs]
                    for k, vs in (item.get("allowed") or {}).items()
                },
            )
        )
    return samples


@dataclass
class CaseResult:
    ok: bool
    detail: str
    latency: float
    subscores: dict[str, float] = field(default_factory=dict)


def eval_analyze_case(client, model: str, sample: ObsSample, *, timeout: float = 150.0) -> CaseResult:
    """One decoration call, scored on the contract analyze_frame's callers need."""
    raw = sample.photo.read_bytes()
    framed = annotate_for_vlm(raw, sample.detections)
    t0 = time.time()
    try:
        analysis = analyze_frame(
            client, model, framed, sample.labels, max_dim=None, timeout_seconds=timeout
        )
    except Exception as exc:
        return CaseResult(False, f"ERROR {exc}", time.time() - t0)
    latency = time.time() - t0

    problems: list[str] = []
    sub: dict[str, float] = {}
    scene = str(analysis.get("scene", "")).strip()
    birds = analysis.get("birds") or []
    by_label = {b["label"]: b for b in birds}

    # Contentless output recycles forever through the backfill — hard problem.
    has_content = bool(scene) or bool(birds)
    sub["content"] = 1.0 if has_content else 0.0
    if not has_content:
        problems.append("contentless analysis")

    # Coverage: one entry per given label (analyze_frame already dropped
    # out-of-set labels, so extras show up as missing coverage only).
    covered = sum(1 for label in sample.labels if label in by_label)
    sub["coverage"] = covered / len(sample.labels)
    if covered < len(sample.labels):
        problems.append(f"covered {covered}/{len(sample.labels)} labels")

    # Scene leakage: overlay/species words, and it should read as a note.
    if scene:
        if checks.contains_overlay_word(scene):
            problems.append("scene mentions the overlay boxes")
            sub["no_overlay"] = 0.0
        else:
            sub["no_overlay"] = 1.0
        if checks.contains_species_word(scene):
            problems.append("scene uses a species word")
            sub["no_species"] = 0.0
        else:
            sub["no_species"] = 1.0
        sub["names_birds"] = 1.0 if checks.mentions_any(scene, sample.labels) else 0.0

    # Evasion: 'hidden'/'unknown' for a bird the detector clearly saw. The
    # incumbent uses these rarely; a model that shrugs on every bird decorates
    # nothing worth keeping.
    activities = [str(b.get("activity", "")) for b in birds]
    evasive = sum(1 for a in activities if a in ("hidden", "unknown"))
    sub["evasion"] = 1.0 - (evasive / len(sample.labels))

    # Activity agreement vs golden (preferred) or silver reference.
    agree_n = agree_hit = 0
    for label in sample.labels:
        got = str(by_label.get(label, {}).get("activity", "")).lower()
        if label in sample.golden:
            agree_n += 1
            agree_hit += 1 if got in sample.golden[label] else 0
        elif label in sample.silver:
            agree_n += 1
            ref = sample.silver[label]
            same_class = ACTIVITY_CLASSES.get(got) == ACTIVITY_CLASSES.get(ref)
            agree_hit += 1 if (got == ref or same_class) else 0
    if agree_n:
        sub["agreement"] = agree_hit / agree_n

    ok = not problems and sub.get("agreement", 1.0) >= 0.5
    label_report = ", ".join(
        f"{label}={by_label.get(label, {}).get('activity', '—')}" for label in sample.labels
    )
    return CaseResult(ok, f"[{sample.camera}] {label_report} :: {scene[:90]}", latency, sub)


def sample_sightings(limit: int = 12):
    """Deterministic spread of collect sidecar frames across distinct birds."""
    by_bird: dict[str, list] = {}
    for json_path in sorted(glob.glob("data/server/collect/**/*.json", recursive=True), reverse=True):
        sighting = _parse_sidecar(Path(json_path))
        if sighting is None or not sighting.width:
            continue
        by_bird.setdefault(sighting.label, []).append(sighting)

    rng = random.Random(SAMPLE_SEED)
    chosen = []
    birds = sorted(by_bird)
    while len(chosen) < limit and birds:
        for bird in list(birds):
            pool = by_bird[bird]
            if not pool:
                birds.remove(bird)
                continue
            chosen.append(pool.pop(rng.randrange(len(pool))))
            if len(chosen) >= limit:
                break
    return chosen


def eval_scene_case(client, model: str, sighting, *, timeout: float = 120.0) -> CaseResult:
    pronouns = pronoun_map(load_sexes())
    context = build_detection_context(
        [_BoxDetection(sighting.label, sighting.bbox)],
        sighting.width, sighting.height, None, pronouns,
    )
    t0 = time.time()
    try:
        caption = describe_scene(
            client, model, sighting.path.read_bytes(),
            context=context, timeout_seconds=timeout,
        )
    except Exception as exc:
        return CaseResult(False, f"ERROR {exc}", time.time() - t0)
    latency = time.time() - t0

    problems = []
    sub: dict[str, float] = {}
    low = caption.lower()
    # A grounded caption names the detected bird (species labels like
    # "cockatiel" are themselves the label word — allowed for group labels).
    from lib.labels import pretty

    name = pretty(sighting.label).lower()
    sub["names_bird"] = 1.0 if name.split()[0] in low else 0.0
    if not sub["names_bird"]:
        problems.append(f"never names {name}")
    sub["concrete"] = 1.0 if any(w in low for w in _ACTIVITY_WORDS) else 0.0
    if not sub["concrete"]:
        problems.append("no concrete activity word")
    words = checks.word_count(caption)
    sub["length"] = 1.0 if words <= 32 else 0.0
    if words > 32:
        problems.append(f"{words} words (prompt asks <25)")
    # The detector context guarantees a bird — "no bird visible" is a refusal.
    if re.search(r"no birds?\b.*(visible|present|seen)|not? clearly visible", low) and "hard to see" not in low:
        problems.append("claims no bird despite detection")
        sub["not_refused"] = 0.0
    else:
        sub["not_refused"] = 1.0
    return CaseResult(not problems, f"[{sighting.label}] {caption[:110]}", latency, sub)


# Concrete activity vocabulary for scene captions (superset of the
# activity_harness word list; kept local so evals don't import the harness).
_ACTIVITY_WORDS = {
    "eat", "eating", "ate", "feed", "feeding", "seed", "seeds", "food", "bowl",
    "perch", "perched", "perching", "preen", "preening", "groom", "grooming",
    "play", "playing", "toy", "bell", "drink", "drinking", "water", "bath",
    "bathing", "splash", "sleep", "sleeping", "nap", "napping", "rest", "resting",
    "fly", "flying", "climb", "climbing", "hang", "hanging", "sit", "sitting",
    "stand", "standing", "watch", "watching", "looking", "chew", "chewing",
    "cage", "branch", "wing", "wings", "alert", "still", "calm",
}


def sample_camera_frames(limit_cameras: int = 6, frames_each: int = 2) -> dict[str, list[Path]]:
    """Distinct camera views from the memories images (IP is in the filename)."""
    by_ip: dict[str, list[Path]] = {}
    for path in sorted(Path("data/server/memories/images").glob("*.jpg"), reverse=True):
        m = re.search(r"camera-([0-9.]+)\.jpg$", path.name)
        if not m:
            continue
        by_ip.setdefault(m.group(1), []).append(path)
    rng = random.Random(SAMPLE_SEED)
    chosen: dict[str, list[Path]] = {}
    for ip in sorted(by_ip, key=lambda ip: -len(by_ip[ip]))[:limit_cameras]:
        pool = by_ip[ip]
        picks = sorted(rng.sample(range(len(pool)), min(frames_each, len(pool))))
        chosen[ip] = [pool[i] for i in picks]
    return chosen


def eval_camera_names(client, model: str, *, limit_cameras: int = 6, timeout: float = 90.0):
    """Name every sampled view; score emptiness, format, uniqueness, stability."""
    frames = sample_camera_frames(limit_cameras=limit_cameras)
    results: list[CaseResult] = []
    names_by_ip: dict[str, list[str]] = {}
    for ip, paths in frames.items():
        for path in paths:
            t0 = time.time()
            try:
                name = name_camera_view(client, model, path.read_bytes(), timeout_seconds=timeout)
            except Exception as exc:
                results.append(CaseResult(False, f"[{ip}] ERROR {exc}", time.time() - t0))
                continue
            latency = time.time() - t0
            problems = []
            if not name:
                problems.append("empty after cleaning")
            elif len(name.split()) > 2:
                problems.append("more than 2 words")
            names_by_ip.setdefault(ip, []).append(name)
            results.append(
                CaseResult(not problems, f"[{ip}] {name!r}", latency, {"named": 0.0 if not name else 1.0})
            )
    # Stability: the same camera should get the same name across frames.
    stable = sum(
        1 for names in names_by_ip.values()
        if len({n.lower() for n in names if n}) <= 1
    )
    # Uniqueness: distinct cameras should not all collapse onto one name.
    all_names = [n.lower() for names in names_by_ip.values() for n in names if n]
    unique_rate = len(set(all_names)) / max(1, len(names_by_ip))
    return results, {
        "stability": stable / max(1, len(names_by_ip)),
        "uniqueness": min(1.0, unique_rate),
    }
