"""Application configuration.

Cameras are no longer hardcoded or supplied as full RTSP URLs. Instead we store
ONLY the Tapo camera-account ``user:password`` (in ``TAPO_CREDENTIALS``) and let
the discovery module sweep the local subnet for cameras whose :554 port is open
and whose ``/stream1`` RTSP path authenticates. This keeps secrets minimal (no
embedded URLs/IPs that drift as DHCP reassigns leases) and makes the camera set
dynamic: the supervisor can add cameras at runtime via the ``/discover`` command.

The RTSP port and stream path are deliberately PURE Python configuration on
``DiscoveryConfig`` (not env vars): they are stable properties of the camera
firmware, not deployment secrets, so baking them in keeps the env surface small.
The only discovery knob exposed to the environment is ``TAPO_DISCOVERY_CIDR``,
which overrides the auto-detected /24 subnet for awkward setups (Docker bridges,
multi-subnet LANs) where the host's primary IP is not on the camera network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    name: str
    enabled: bool
    rtsp_url: str
    # The camera's IP address. Discovery fills this in so the supervisor can use
    # it as a stable identity (dedup across rediscovery, since DHCP usually keeps
    # leases) and so the dashboard can show the host instead of a generic name.
    host: str = ""
    sample_fps: float = 1.0
    # Base reconnect delay; grows exponentially up to ``max_reconnect_seconds``
    # while a camera stays unreachable, then resets once frames flow again.
    reconnect_seconds: float = 5.0
    max_reconnect_seconds: float = 60.0
    # Hard caps so a missing camera can't block a worker thread forever. Tapo
    # streams are flaky, so opening and reading must be able to give up.
    open_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 15.0
    # RTSP over TCP is far more reliable than the UDP default on lossy WiFi.
    rtsp_transport: str = "tcp"


@dataclass(frozen=True)
class ModelConfig:
    # One or more model files; each runs a separate pass over every frame and
    # their detections are merged. A tuple (not a list) so this frozen dataclass
    # stays hashable.
    paths: tuple[Path, ...]
    # The detection confidence floor. 0.5 is the F1 peak on the held-out test
    # split; the live server can override it via the MODEL_CONFIDENCE env var.
    confidence: float = 0.5
    iou: float = 0.5
    image_size: int = 960
    device: str = "auto"


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: str
    user_ids: list[str]
    last_seen_alert_seconds: float = 900.0
    bbox_movement_alert_ratio: float = 0.10
    include_snapshot: bool = True


@dataclass(frozen=True)
class OllamaConfig:
    # Points at the Ollama already running on the machine — this project never
    # installs or manages Ollama, it only talks to it over HTTP. Two models: a
    # language model for intent parsing + chat, and a vision model for captioning
    # bird photos. Both default to models the user already has pulled.
    enabled: bool
    base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:4b"
    vlm_model: str = "qwen2.5vl:7b"
    # Generous: vision passes over a photo can take tens of seconds on a busy GPU.
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class CollectConfig:
    objects: frozenset[str]
    directory: Path = Path("./data/server/collect")


@dataclass(frozen=True)
class FilterConfig:
    # When non-empty, only these object labels are allowed to alert and are the
    # only labels surfaced in the terminal stats (detect row + objects panel);
    # everything else is still detected/collected but never triggers a
    # notification and is kept out of the display. Empty means no filtering
    # (alert on and show every detected object).
    objects: frozenset[str]


@dataclass(frozen=True)
class CameraCredentials:
    # The Tapo camera-account user/password. The same credentials work for every
    # camera on the LAN, so we store them once and reuse them for each
    # discovered host when building its RTSP URL.
    username: str
    password: str


@dataclass(frozen=True)
class DiscoveryConfig:
    # The RTSP port and stream path are firmware constants, not secrets, so they
    # live here as pure .py config rather than as env vars.
    rtsp_port: int = 554
    stream_path: str = "/stream1"
    # When None, discovery auto-detects the host's primary IPv4 and scans its
    # /24. Set ``cidr`` (or the TAPO_DISCOVERY_CIDR env var) to scan a different
    # subnet, e.g. inside Docker where the host IP is on a bridge network.
    cidr: str | None = None
    # An explicit host list. When non-empty it overrides cidr/auto-detect; tests
    # use this to point the sweep at a localhost fake-RTSP server.
    hosts: tuple[str, ...] = ()
    # Concurrency of the TCP sweep. A /24 has 254 hosts, so a wide pool lets the
    # whole subnet finish in roughly one connect-timeout instead of serially.
    max_workers: int = 256
    # Per-host TCP probe of :554. Kept short because most addresses are dead and
    # we don't want the sweep to stall on each unanswered SYN.
    connect_timeout_seconds: float = 0.5
    # Per-host RTSP DESCRIBE handshake. Generous relative to the TCP probe since
    # only the (few) hosts with :554 open reach this stage.
    rtsp_timeout_seconds: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    snapshot_dir: Path
    model: ModelConfig
    telegram: TelegramConfig
    collect: CollectConfig
    filter: FilterConfig
    credentials: CameraCredentials
    discovery: DiscoveryConfig
    # Defaulted (and last) so existing constructions that predate the AI features
    # — and tests — don't have to pass them. ``build_config`` always sets them.
    ollama: OllamaConfig = OllamaConfig(enabled=False)
    # The caretaker's report beat in minutes (0 disables). It checks far more
    # often than this, reporting new birds immediately and only editing the last
    # message in place on this beat when nothing changed. When reports are on, raw
    # per-detection photo alerts default OFF so they don't double up.
    memory_interval_minutes: float = 0.0
    raw_photo_alerts: bool = True
    # /find PTZ patrol scans this grid of each camera's range (columns x rows),
    # so it sweeps the tilt axis too. 4 x 3 = 12 cells by default.
    ptz_scan_cols: int = 4
    ptz_scan_rows: int = 3


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _as_user_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_object_names(value: str) -> frozenset[str]:
    return frozenset(item.strip().lower() for item in value.split(",") if item.strip())


def _as_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _as_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _credentials() -> CameraCredentials:
    raw = _require_env("TAPO_CREDENTIALS")  # "user:password"
    # ``partition`` splits on the FIRST colon only, so any colons in the password
    # stay in the password (passwords may legitimately contain ':').
    user, sep, password = raw.partition(":")
    if not sep or not user:
        raise ValueError("TAPO_CREDENTIALS must be in 'user:password' form")
    return CameraCredentials(username=user, password=password)


def _model_paths() -> tuple[Path, ...]:
    raw = _require_env("MODEL_PATH")
    paths = tuple(Path(item.strip()) for item in raw.split(",") if item.strip())
    if not paths:
        raise ValueError("No model paths configured; set MODEL_PATH")
    return paths


def _model_confidence() -> float:
    """Detection confidence floor; the env overrides the ModelConfig default.

    Kept as an env knob (unlike iou/image_size) because the operating point is a
    deployment trade-off — lower it for more recall, raise it for more precision —
    that you tune without editing code. Blank/unset falls back to the single
    source of truth, the dataclass default.
    """
    raw = os.environ.get("MODEL_CONFIDENCE", "").strip()
    if not raw:
        return ModelConfig.confidence
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"MODEL_CONFIDENCE must be a number, got {raw!r}") from exc


def _model_image_size() -> int:
    """Inference image size (px); the env overrides the ModelConfig default.

    Like confidence, this is a deployment trade-off worth exposing: higher sizes
    recover recall on small/distant birds (live-007 measured best at 1280) at the
    cost of slower inference. Blank/unset falls back to the dataclass default.
    """
    raw = os.environ.get("MODEL_IMAGE_SIZE", "").strip()
    if not raw:
        return ModelConfig.image_size
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"MODEL_IMAGE_SIZE must be an integer, got {raw!r}") from exc


def _ollama_config() -> OllamaConfig:
    """Build the Ollama client config from the environment.

    Enabled by default (the whole point is to talk to the machine's existing
    Ollama); set ``OLLAMA_ENABLED=0`` to turn the natural-language features off.
    Blank model/url envs fall back to the dataclass defaults — the single source
    of truth — so an empty value in .env behaves like "unset".
    """
    enabled = os.environ.get("OLLAMA_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or OllamaConfig.base_url
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "").strip() or OllamaConfig.llm_model
    vlm_model = os.environ.get("OLLAMA_VLM_MODEL", "").strip() or OllamaConfig.vlm_model
    return OllamaConfig(
        enabled=enabled, base_url=base_url, llm_model=llm_model, vlm_model=vlm_model
    )


def build_config() -> AppConfig:
    model = ModelConfig(
        paths=_model_paths(),
        confidence=_model_confidence(),
        image_size=_model_image_size(),
    )

    telegram = TelegramConfig(
        enabled=True,
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        user_ids=_as_user_ids(os.environ.get("TELEGRAM_USER_IDS", "")),
        include_snapshot=True,
    )
    collect = CollectConfig(objects=_as_object_names(os.environ.get("COLLECT_OBJECTS", "")))
    object_filter = FilterConfig(objects=_as_object_names(os.environ.get("FILTER_OBJECTS", "")))

    credentials = _credentials()
    # Start from the pure-.py defaults, then let an explicit env override the
    # subnet when auto-detect would pick the wrong interface (Docker, etc.).
    discovery = DiscoveryConfig(cidr=os.environ.get("TAPO_DISCOVERY_CIDR") or None)

    # Caretaker report beat, and whether raw photo alerts still fire. When the
    # caretaker is on, the raw per-detection photo spam is off by default (it
    # reports the photos instead); RAW_PHOTO_ALERTS overrides either way.
    memory_minutes = _as_float("MEMORY_INTERVAL_MINUTES", 5.0)
    raw_default = memory_minutes <= 0
    raw_photo_alerts = _as_bool("RAW_PHOTO_ALERTS", raw_default)

    return AppConfig(
        snapshot_dir=Path("./data/server/snapshots"),
        model=model,
        telegram=telegram,
        collect=collect,
        filter=object_filter,
        credentials=credentials,
        discovery=discovery,
        ollama=_ollama_config(),
        memory_interval_minutes=memory_minutes,
        raw_photo_alerts=raw_photo_alerts,
        ptz_scan_cols=max(1, int(_as_float("PTZ_SCAN_COLS", 4))),
        ptz_scan_rows=max(1, int(_as_float("PTZ_SCAN_ROWS", 3))),
    )
