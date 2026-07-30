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
    # MUST be an INSTRUCT model, not a reasoning one. A reasoning model (qwen3) spends
    # its whole num_predict budget "thinking" and returns empty chat/activity replies
    # (the "Sorry — I garbled that" fallback), or leaks that reasoning into the answer.
    # gemma3:4b answers directly, stays terse, and routes intents just as accurately.
    llm_model: str = "gemma3:4b"
    # The model for ACTIVITY/MEMORY recall (day/week summaries, "who was most active",
    # "any bird didn't eat", co-occurrence). This reasons over per-bird tallies, which
    # a 4B model gets wrong (it confabulates); a bigger instruct model (gemma3:12b) is
    # markedly more accurate. Kept separate so intent/chat stay on the fast 4B. Falls
    # back to llm_model when unset.
    recall_model: str = "gemma3:12b"
    # 3B (not 7B): on an 8GB GPU the 7B VLM (4.75GB) + LLM (3.2GB) + YOLO spilled
    # to CPU. The 3B is 2.16GB at the same ~20s latency and near-identical caption
    # quality, so all three coexist on-GPU (~5.8GB) with headroom. See bench.
    vlm_model: str = "qwen2.5vl:3b"
    # Generous: vision passes over a photo can take tens of seconds on a busy GPU.
    timeout_seconds: float = 120.0
    # Max concurrent vision (image) calls; the rest queue. 1 keeps the GPU from
    # being swamped when many frames are captioned/named at once.
    vision_concurrency: int = 1
    # Ollama keep_alive: how long a worker keeps a model loaded after a request
    # ("30m", "1h", "-1" = forever). This server's calls are sporadic (a chat
    # turn, a 5-minute memory beat) — longer than Ollama's 5m default, so
    # without this nearly every call pays a cold model load first.
    keep_alive: str = "30m"


@dataclass(frozen=True)
class PrivacyConfig:
    # Withhold any outbound TELEGRAM photo that shows a person, so the owners
    # never broadcast themselves walking through the aviary frame. The screen
    # runs only on the Telegram upload path — local snapshots, collection and
    # memory photos are untouched — and the alert/report text still goes out,
    # with a note explaining the missing photo.
    enabled: bool = True
    # Must be a model that knows a 'person' class. The trained bird models
    # don't, so this is a stock COCO model; nano is plenty — a person in frame
    # is large, and the screen is recall-biased anyway.
    model_path: Path = Path("./yolo11n.pt")
    # LOW on purpose: a missed person leaks a photo of us to Telegram, while a
    # false positive merely withholds one bird photo. Recall over precision.
    confidence: float = 0.25
    image_size: int = 640
    # CPU so the screen never competes for the GPU's headroom (YOLO + VLM + LLM
    # already share it); one nano pass per outbound photo is cheap.
    device: str = "cpu"


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
    # Concurrency of the sweep. Each worker runs the full RTSP DESCRIBE handshake
    # against its host (there is no cheap-prefilter stage), so the cap is kept low:
    # fanning all 254 of a /24 out at once overwhelms tiny WiFi devices and the LAN,
    # making the sweep flaky. A small pool probes hosts in steady waves instead.
    max_workers: int = 8
    # Per-host TCP probe of :554. Kept short because most addresses are dead and
    # we don't want the sweep to stall on each unanswered SYN.
    connect_timeout_seconds: float = 0.5
    # Per-host RTSP DESCRIBE read/auth timeout after the TCP connection opens.
    rtsp_timeout_seconds: float = 3.0
    # Discovery is best-effort against tiny WiFi devices that may briefly reject
    # or stall an RTSP socket while existing streams are active. Retry transient
    # connect/protocol failures a few times before declaring the host absent.
    probe_attempts: int = 3
    probe_retry_delay_seconds: float = 0.15


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
    privacy: PrivacyConfig = PrivacyConfig()
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
    # Location for sunrise/sunset and the weather outlook. NO real coordinates
    # live in the repo — set AVIARY_LATITUDE / AVIARY_LONGITUDE in .env. The 0,0
    # placeholder means "unset": sun times fall back to clock anchors and the
    # weather feature stays idle until a location is configured.
    latitude: float = 0.0
    longitude: float = 0.0
    # UTC offset (hours) for sunrise/sunset local times — set AVIARY_UTC_OFFSET_HOURS
    # in .env. 0 = UTC. (Assumes no DST, which suits a fixed-offset deployment.)
    utc_offset_hours: float = 0.0
    # Weather outlook + bird-safety warnings (Open-Meteo, no API key; needs a
    # location set above). On by default; WEATHER_ALERTS=0 turns off the proactive
    # hot-day / cold-night warnings (the /weather command still works).
    weather_alerts: bool = True
    # Bird-safety thresholds in °C: warn when the day's high reaches hot_c or the
    # night's low drops to cold_c. Defaults suit tropical cage birds.
    weather_hot_c: float = 32.0
    weather_cold_c: float = 10.0
    # The optional post-wake "how the birds slept" report. Off by default — the
    # /sleep command and the /status night-in-progress line work regardless;
    # SLEEP_MORNING_REPORT=1 also pushes a summary each morning when they wake.
    sleep_morning_report: bool = False
    # The Tapo CLOUD account password (what the Tapo app logs in with). Unlocks
    # the cameras' proprietary local API (lib.tapo) for what ONVIF can't reach:
    # the /flash spotlight command, automatic spotlights during a night /find,
    # and rebooting a camera whose stream has wedged. The camera-account
    # credentials above CANNOT authenticate that API. Empty = features dormant.
    tapo_cloud_password: str = ""
    # RGB status display (lib.rgb): drive the motherboard / header / strip LEDs to
    # mirror aviary state — a loading-wave during discovery, each bird's color on
    # detection, a single LED at night. Off by default; needs a running OpenRGB
    # SDK server (see scripts/install-rgb.sh). rgb_strip_len sets the pixel count of an empty
    # addressable header (e.g. an ARGB fan); rgb_night_led picks the night LED
    # (-1 = auto = last LED). Per-LED→bird mapping lives in data/server/rgb_layout.json
    # (write it with `uv run rgb-calibrate`).
    # rgb_strip_len: -1 = leave the addressable header at OpenRGB's detected size;
    # 0 = force it OFF (disable the ARGB header, e.g. back to onboard-only);
    # N = set N pixels (the count of the ARGB strip/fan plugged into the header).
    rgb_enabled: bool = False
    rgb_host: str = "127.0.0.1"
    rgb_port: int = 6742
    rgb_brightness: float = 1.0
    rgb_strip_len: int = -1
    rgb_night_led: int = -1


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
    # One endpoint for both: a direct Ollama (http://host:11434) OR an Olla load
    # balancer that spreads calls across several Ollama machines (use Olla's
    # Ollama path, e.g. http://host:40114/olla/ollama). Both speak the same
    # /api/chat + /api/generate API, so a single base URL covers either.
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or OllamaConfig.base_url
    llm_model = os.environ.get("OLLAMA_LLM_MODEL", "").strip() or OllamaConfig.llm_model
    vlm_model = os.environ.get("OLLAMA_VLM_MODEL", "").strip() or OllamaConfig.vlm_model
    recall_model = (
        os.environ.get("OLLAMA_RECALL_MODEL", "").strip()
        or OllamaConfig.recall_model
        or llm_model
    )
    vision_concurrency = max(1, int(_as_float("OLLAMA_VISION_CONCURRENCY", OllamaConfig.vision_concurrency)))
    timeout_seconds = _as_float("OLLAMA_TIMEOUT_SECONDS", OllamaConfig.timeout_seconds)
    keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "").strip() or OllamaConfig.keep_alive
    return OllamaConfig(
        enabled=enabled, base_url=base_url, llm_model=llm_model, vlm_model=vlm_model,
        recall_model=recall_model, vision_concurrency=vision_concurrency,
        timeout_seconds=timeout_seconds, keep_alive=keep_alive,
    )


def _privacy_config() -> PrivacyConfig:
    """Build the Telegram privacy screen config from the environment.

    On by default — photos of people should never leave the machine unless the
    owner explicitly opts out with ``PRIVACY_FILTER=0``. Blank envs fall back to
    the dataclass defaults, the single source of truth.
    """
    model_path = os.environ.get("PRIVACY_MODEL_PATH", "").strip()
    return PrivacyConfig(
        enabled=_as_bool("PRIVACY_FILTER", PrivacyConfig.enabled),
        model_path=Path(model_path) if model_path else PrivacyConfig.model_path,
        confidence=_as_float("PRIVACY_CONFIDENCE", PrivacyConfig.confidence),
    )


def _model_device() -> str:
    """Detector device; the env overrides the ModelConfig default ("auto").

    "auto" picks the GPU with the most free VRAM on multi-GPU boxes (see
    detector.py); set e.g. MODEL_DEVICE=cuda:1 or cpu to pin it by hand.
    """
    return os.environ.get("MODEL_DEVICE", "").strip() or ModelConfig.device

def build_config() -> AppConfig:
    model = ModelConfig(
        paths=_model_paths(),
        confidence=_model_confidence(),
        image_size=_model_image_size(),
        device=_model_device(),
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

    # Caretaker report beat, and whether raw photo alerts still fire. The
    # caretaker only runs when the interval is set AND Ollama is on; raw alerts
    # default OFF only when the caretaker will actually replace them, so a config
    # with Ollama disabled (or no interval) keeps the raw alerts and never goes
    # silent. RAW_PHOTO_ALERTS overrides either way.
    ollama = _ollama_config()
    memory_minutes = _as_float("MEMORY_INTERVAL_MINUTES", 5.0)
    caretaker_on = memory_minutes > 0 and ollama.enabled
    raw_photo_alerts = _as_bool("RAW_PHOTO_ALERTS", not caretaker_on)

    return AppConfig(
        snapshot_dir=Path("./data/server/snapshots"),
        model=model,
        telegram=telegram,
        collect=collect,
        filter=object_filter,
        credentials=credentials,
        discovery=discovery,
        ollama=ollama,
        privacy=_privacy_config(),
        memory_interval_minutes=memory_minutes,
        raw_photo_alerts=raw_photo_alerts,
        ptz_scan_cols=max(1, int(_as_float("PTZ_SCAN_COLS", 4))),
        ptz_scan_rows=max(1, int(_as_float("PTZ_SCAN_ROWS", 3))),
        latitude=_as_float("AVIARY_LATITUDE", 0.0),
        longitude=_as_float("AVIARY_LONGITUDE", 0.0),
        utc_offset_hours=_as_float("AVIARY_UTC_OFFSET_HOURS", 0.0),
        weather_alerts=_as_bool("WEATHER_ALERTS", True),
        weather_hot_c=_as_float("WEATHER_HOT_C", 32.0),
        weather_cold_c=_as_float("WEATHER_COLD_C", 10.0),
        sleep_morning_report=_as_bool("SLEEP_MORNING_REPORT", False),
        tapo_cloud_password=os.environ.get("TAPO_CLOUD_PASSWORD", "").strip(),
        rgb_enabled=_as_bool("RGB_ENABLED", False),
        rgb_host=os.environ.get("RGB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        rgb_port=int(_as_float("RGB_PORT", 6742)),
        rgb_brightness=_as_float("RGB_BRIGHTNESS", 1.0),
        rgb_strip_len=int(_as_float("RGB_STRIP_LEN", -1)),
        rgb_night_led=int(_as_float("RGB_NIGHT_LED", -1)),
    )
