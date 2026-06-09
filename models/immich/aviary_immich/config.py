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
    """CLIP is opt-in: it needs ``open_clip`` installed and a calibrated threshold.

    Off by default so a stock run needs no extra dependency and stays behavior-identical; the Tennis
    album still works via the YOLO ``tennis racket`` signal alone. Turn it on with ``IMMICH_CLIP=1``
    after ``uv sync --group clip`` and calibrating the cosine threshold.
    """
    return os.getenv("IMMICH_CLIP", "0").strip().lower() in {"1", "true", "on", "yes"}


# The models that produce signals, and the rules that turn those signals into albums. YOLO detects
# objects (bird/dog/cat/tennis racket); CLIP (opt-in) detects scenes (tennis court). Tennis unions
# the two so either a racket OR a court files a photo. Agreement (second-opinion) mode lands in
# phase 3 by flipping a rule's mode/min_votes.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(name="yolo", kind="yolo", labels=("bird", "dog", "cat", "tennis racket")),
    ModelSpec(
        name="clip",
        kind="clip",
        prompts={
            "tennis court": ("a tennis court", "people playing tennis", "a tennis match"),
        },
        # Cosine similarity, NOT YOLO's 0.30 scale. Calibrated on sample images (ViT-B-32 laion2b):
        # a real tennis court scored ~0.24, while "tennis court" on non-court photos stayed ~0.07,
        # so 0.22 captures courts with a wide margin. Re-tune on your own library if you see noise.
        threshold=0.22,
        enabled=_clip_enabled(),
        options={"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
    ),
)

ALBUM_RULES: tuple[AlbumRule, ...] = (
    AlbumRule("Birds", (Signal("yolo", "bird"),), mode="union"),
    AlbumRule("Dogs", (Signal("yolo", "dog"),), mode="union"),
    AlbumRule("Cats", (Signal("yolo", "cat"),), mode="union"),
    AlbumRule("Tennis", (Signal("yolo", "tennis racket"), Signal("clip", "tennis court")), mode="union"),
)


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
