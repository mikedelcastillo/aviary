"""Config and .env loading for Immich scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aviary_immich.rules import AlbumRule, Signal


BIRD_ALBUM_NAME = "Birds"


@dataclass(frozen=True)
class ModelSpec:
    """Declares one model the pipeline runs and the tags it can emit.

    ``kind`` selects the backend (``"yolo"`` object detector / ``"clip"`` scene classifier).
    ``labels`` are the COCO class names a YOLO model keeps. ``prompts`` map a canonical tag to the
    CLIP text prompts that detect it. ``threshold=None`` means "use the CLI ``--threshold``" (the
    YOLO default); CLIP sets its own cosine threshold because the scales differ. ``options`` carries
    kind-specific extras (e.g. CLIP ``model_name``/``pretrained``).
    """

    name: str
    kind: str
    labels: tuple[str, ...] = ()
    prompts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    threshold: float | None = None
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


def _clip_enabled() -> bool:
    """Whether the CLIP scene model runs (court/selfie signals + the animal second opinion).

    ``IMMICH_CLIP`` forces it: ``1``/``on`` enables, ``0``/``off`` disables. The default is ``auto``:
    enabled wherever ``open_clip`` is importable, otherwise it gracefully falls back to YOLO-only so
    a machine without the CLIP stack still runs (rather than crashing on the missing import). Install
    it with ``scripts/install-clip.{sh,ps1}`` (out-of-band, like torch) to turn it on by default.
    """
    override = os.getenv("IMMICH_CLIP", "auto").strip().lower()
    if override in {"0", "false", "off", "no"}:
        return False
    if override in {"1", "true", "on", "yes"}:
        return True
    import importlib.util

    return importlib.util.find_spec("open_clip") is not None


def model_specs(clip_enabled: bool) -> tuple[ModelSpec, ...]:
    """The models to run. YOLO detects objects; CLIP (opt-in) detects scenes.

    ``person`` is only consumed by the Selfies second-opinion, so YOLO only bothers detecting it
    when CLIP is on. The CLIP spec is always present but ``enabled`` tracks the flag.
    """
    yolo_labels = ("bird", "dog", "cat", "tennis racket") + (("person",) if clip_enabled else ())
    return (
        ModelSpec(name="yolo", kind="yolo", labels=yolo_labels),
        ModelSpec(
            name="clip",
            kind="clip",
            prompts={
                "bird": ("a photo of a bird", "a bird"),
                "dog": ("a photo of a dog", "a dog"),
                "cat": ("a photo of a cat", "a cat"),
                "tennis court": ("a tennis court", "people playing tennis", "a tennis match"),
                "selfie": ("a selfie", "a person taking a selfie", "a close-up self-portrait photo"),
            },
            # Cosine similarity, NOT YOLO's 0.30 scale. Calibrated on sample images (ViT-B-32
            # laion2b): correct-class scores landed ~0.24-0.30 (bird 0.295, dog 0.274, court 0.236,
            # selfie 0.244) while wrong-class scores stayed <0.21, so 0.22 separates them. Re-tune
            # on your own library if you see noise.
            threshold=0.22,
            enabled=clip_enabled,
            options={"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
        ),
    )


def _animal_rule(album: str, tag: str, clip_enabled: bool) -> AlbumRule:
    """YOLO alone (union) by default; a YOLO+CLIP second opinion (agreement) when CLIP is on.

    Agreement = both models must independently see the animal, raising precision at some recall
    cost. It needs the second model, so it only engages under ``IMMICH_CLIP=1``; otherwise the album
    falls back to YOLO-only union so a stock run still fills normally.
    """
    if clip_enabled:
        return AlbumRule(album, (Signal("yolo", tag), Signal("clip", tag)), mode="agreement", min_votes=2)
    return AlbumRule(album, (Signal("yolo", tag),), mode="union")


def album_rules(clip_enabled: bool) -> tuple[AlbumRule, ...]:
    """The album rules, adapted to whether the CLIP second model is available."""
    rules: list[AlbumRule] = [
        _animal_rule("Birds", "bird", clip_enabled),
        _animal_rule("Dogs", "dog", clip_enabled),
        _animal_rule("Cats", "cat", clip_enabled),
        AlbumRule("Tennis", (Signal("yolo", "tennis racket"), Signal("clip", "tennis court")), mode="union"),
    ]
    if clip_enabled:
        # A selfie has no distinct object, so require BOTH a person (YOLO) and a selfie composition
        # (CLIP) — that keeps group shots and scenery that merely score high on "selfie" out. A
        # CLIP-only concept, so the album exists only when CLIP is enabled.
        rules.append(
            AlbumRule("Selfies", (Signal("yolo", "person"), Signal("clip", "selfie")), mode="agreement", min_votes=2)
        )
    return tuple(rules)


MODELS: tuple[ModelSpec, ...] = model_specs(_clip_enabled())
ALBUM_RULES: tuple[AlbumRule, ...] = album_rules(_clip_enabled())


def yolo_labels(models: tuple[ModelSpec, ...] = MODELS) -> tuple[str, ...]:
    """COCO labels every enabled YOLO model needs (union across specs)."""
    labels: list[str] = []
    for spec in models:
        if spec.enabled and spec.kind == "yolo":
            for label in spec.labels:
                if label not in labels:
                    labels.append(label)
    return tuple(labels)


def album_names(rules: tuple[AlbumRule, ...] = ALBUM_RULES) -> tuple[str, ...]:
    """Album names in declaration order, de-duplicated."""
    names: list[str] = []
    for rule in rules:
        if rule.album_name not in names:
            names.append(rule.album_name)
    return tuple(names)


# Back-compat shim: kept so any external caller/import still resolves. No longer the routing source
# (albums are driven by ALBUM_RULES). Derived from the YOLO model + rules to stay consistent.
ANIMAL_ALBUMS = {
    signal.tag: rule.album_name
    for rule in ALBUM_RULES
    for signal in rule.signals
    if signal.model == "yolo"
}
ANIMAL_LABELS = yolo_labels()


@dataclass(frozen=True)
class AccountConfig:
    slug: str
    api_key_env: str
    api_key: str
    enabled: bool = True


@dataclass(frozen=True)
class ImmichConfig:
    base_url: str
    accounts: list[AccountConfig]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise ValueError("IMMICH_BASE_URL is empty")
    if not normalized.endswith("/api"):
        normalized = f"{normalized}/api"
    return normalized


def load_accounts_config(path: Path, env_file: Path = Path(".env")) -> ImmichConfig:
    load_env_file(env_file)

    import yaml

    if not path.exists():
        raise FileNotFoundError(
            f"Missing account config: {path}."
        )

    base_url = os.getenv("IMMICH_BASE_URL")
    if not base_url:
        raise ValueError("IMMICH_BASE_URL must be set in .env, for example http://192.168.1.168:2283/api")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    accounts = [_load_account(item) for item in raw.get("accounts", [])]
    enabled_accounts = [account for account in accounts if account.enabled]
    if not enabled_accounts:
        raise ValueError(f"No enabled accounts found in {path}")

    return ImmichConfig(base_url=normalize_base_url(base_url), accounts=enabled_accounts)


def _load_account(raw: dict[str, Any]) -> AccountConfig:
    slug = str(raw["slug"]).strip()
    api_key_env = str(raw["api_key_env"]).strip()
    api_key = os.getenv(api_key_env, "").strip()

    if not slug:
        raise ValueError("Account slug cannot be empty")
    if not api_key_env:
        raise ValueError(f"Account {slug} is missing api_key_env")
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is required for account {slug}")

    return AccountConfig(
        slug=slug,
        api_key_env=api_key_env,
        api_key=api_key,
        enabled=bool(raw.get("enabled", True)),
    )
