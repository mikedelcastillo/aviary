"""Machine telemetry for the ``/machine`` command — CPU, GPU, network, cluster.

Answers "how is the box doing?": which CPU/GPU it is, their usage + temperature,
the network throughput (up/down), and — when the AI endpoint is an Olla load
balancer rather than a single Ollama — the per-node health of the cluster behind
it.

Every changing metric is a **1-minute rolling average**, not a one-shot reading,
so a momentary spike (a YOLO inference burst, a single hot frame) doesn't
dominate the number. A background :class:`MachineSampler` thread samples all the
sources every few seconds and keeps the trailing-minute window warm, so
``/machine`` answers instantly without blocking to measure:

  * CPU usage  -> the busy fraction across the window from two ``/proc/stat`` reads,
  * CPU temp   -> the mean of the window's ``k10temp``/``coretemp`` hwmon readings,
  * GPU usage/temp -> the mean of the window's ``nvidia-smi`` readings (VRAM = latest),
  * network    -> the average byte rate across the window from ``/proc/net/dev``.

Static identity (CPU model + thread count, from ``/proc/cpuinfo``) and the Olla
node health (a point-in-time status, fetched live) aren't averaged.

This needs **no new dependency** — only what a Linux + NVIDIA box already exposes
(``/proc``, ``/sys``, ``nvidia-smi``) plus ``requests``. The parsing / windowing /
formatting are pure functions so they unit-test without touching ``/proc``,
``/sys``, a subprocess, or the network; the ``read_*`` / ``fetch_*`` helpers are
the only parts that do real I/O.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger("lib.machine")


# ---------------------------------------------------------------------------
# CPU usage (/proc/stat) and identity (/proc/cpuinfo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CpuTimes:
    """A snapshot of the aggregate CPU jiffies from ``/proc/stat``."""

    total: int
    idle: int


def parse_cpu_times(text: str) -> CpuTimes | None:
    """Parse the aggregate ``cpu`` line of ``/proc/stat`` into total/idle jiffies.

    The line is ``cpu  user nice system idle iowait irq softirq steal ...``;
    idle time counts both ``idle`` and ``iowait``. Returns None if the line is
    missing or malformed.
    """
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        try:
            nums = [int(part) for part in line.split()[1:]]
        except ValueError:
            return None
        if len(nums) < 4:
            return None
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
        return CpuTimes(total=sum(nums), idle=idle)
    return None


def cpu_usage_percent(prev: CpuTimes, cur: CpuTimes) -> float | None:
    """Busy-fraction over the interval between two ``/proc/stat`` snapshots.

    Returns None when the counters didn't advance (no elapsed time, or a
    counter wrap) so the caller shows "n/a" rather than a bogus number.
    """
    total_delta = cur.total - prev.total
    idle_delta = cur.idle - prev.idle
    if total_delta <= 0 or idle_delta < 0:
        return None
    busy = (total_delta - idle_delta) / total_delta * 100.0
    return max(0.0, min(100.0, busy))


def read_cpu_times(proc_stat: str = "/proc/stat") -> CpuTimes | None:
    try:
        return parse_cpu_times(Path(proc_stat).read_text())
    except OSError:
        LOGGER.warning("Could not read %s", proc_stat)
        return None


def parse_cpu_model(text: str) -> tuple[str | None, int | None]:
    """Parse ``/proc/cpuinfo`` for the CPU model name + logical-thread count.

    The model name is the first ``model name`` line (absent on some ARM SoCs ->
    None); the thread count is the number of ``processor`` entries.
    """
    name: str | None = None
    threads = 0
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        if key == "model name" and name is None:
            name = value.strip()
        elif key == "processor":
            threads += 1
    return name, (threads or None)


def read_cpu_model(proc_cpuinfo: str = "/proc/cpuinfo") -> tuple[str | None, int | None]:
    try:
        return parse_cpu_model(Path(proc_cpuinfo).read_text())
    except OSError:
        return None, None


# ---------------------------------------------------------------------------
# RAM (/proc/meminfo)
# ---------------------------------------------------------------------------


def _meminfo_kb(rest: str) -> float | None:
    try:
        return float(rest.split()[0])  # "32768000 kB" -> 32768000
    except (IndexError, ValueError):
        return None


def parse_meminfo(text: str) -> tuple[float, float] | None:
    """Parse ``/proc/meminfo`` into ``(used_mb, total_mb)``.

    Used is ``MemTotal - MemAvailable`` — the kernel's own estimate of memory in
    active use (the figure ``free`` / ``htop`` show), which already accounts for
    reclaimable cache. Returns None if either field is missing (e.g. a pre-3.14
    kernel without ``MemAvailable``).
    """
    total_kb = avail_kb = None
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        if key == "MemTotal":
            total_kb = _meminfo_kb(rest)
        elif key == "MemAvailable":
            avail_kb = _meminfo_kb(rest)
        if total_kb is not None and avail_kb is not None:
            break
    if total_kb is None or avail_kb is None:
        return None
    used_mb = max(0.0, (total_kb - avail_kb) / 1024)
    return used_mb, total_kb / 1024


def read_meminfo_mb(proc_meminfo: str = "/proc/meminfo") -> tuple[float, float] | None:
    try:
        return parse_meminfo(Path(proc_meminfo).read_text())
    except OSError:
        LOGGER.warning("Could not read %s", proc_meminfo)
        return None


def _mem_percent(used_mb: float | None, total_mb: float | None) -> float | None:
    if used_mb is None or total_mb is None or total_mb <= 0:
        return None
    return max(0.0, min(100.0, used_mb / total_mb * 100.0))


# ---------------------------------------------------------------------------
# CPU temperature (/sys/class/hwmon)
# ---------------------------------------------------------------------------

# hwmon device names that report a CPU package temperature, in preference order
# (AMD Ryzen -> k10temp/zenpower, Intel -> coretemp, ARM -> cpu_thermal). The
# generic ``acpitz`` zone is deliberately NOT here: on many boards it reports a
# bogus low value (~16 °C), so we only fall back to it when nothing better exists.
_CPU_TEMP_NAMES = ("k10temp", "zenpower", "coretemp", "cpu_thermal", "cpu-thermal")
# Within a CPU sensor, the label that best represents the package/die temp.
_CPU_TEMP_LABELS = ("tctl", "tdie", "package id 0", "package", "cpu")


@dataclass(frozen=True)
class HwmonDevice:
    name: str
    # (label, milli-degrees-C) for each tempN_input; label is "" when unlabeled.
    temps: tuple[tuple[str, int], ...]


def _cpu_name_rank(name: str) -> int:
    lowered = name.strip().lower()
    for rank, candidate in enumerate(_CPU_TEMP_NAMES):
        if lowered.startswith(candidate):
            return rank
    return len(_CPU_TEMP_NAMES)


def select_cpu_temp(devices: list[HwmonDevice]) -> float | None:
    """Pick the best CPU temperature (°C) from the hwmon devices, or None.

    Prefers a known CPU sensor (k10temp/coretemp/...) over anything else, and
    within it prefers a package/die-labeled reading (``Tctl``, ``Package id 0``)
    over an arbitrary one. Devices that aren't CPU sensors are ignored entirely.
    """
    cpu_devices = sorted(
        (d for d in devices if d.temps and _cpu_name_rank(d.name) < len(_CPU_TEMP_NAMES)),
        key=lambda d: _cpu_name_rank(d.name),
    )
    for device in cpu_devices:
        for wanted in _CPU_TEMP_LABELS:
            for label, milli in device.temps:
                if wanted in label.strip().lower():
                    return milli / 1000.0
        return device.temps[0][1] / 1000.0
    return None


def read_hwmon_devices(hwmon_root: str = "/sys/class/hwmon") -> list[HwmonDevice]:
    """Read every hwmon device's name + tempN_input/label pairs from sysfs."""
    devices: list[HwmonDevice] = []
    try:
        children = sorted(Path(hwmon_root).iterdir())
    except OSError:
        return devices
    for child in children:
        try:
            name = (child / "name").read_text().strip()
        except OSError:
            continue
        temps: list[tuple[str, int]] = []
        for temp_input in sorted(child.glob("temp*_input")):
            try:
                milli = int(temp_input.read_text().strip())
            except (OSError, ValueError):
                continue
            label_path = Path(str(temp_input).replace("_input", "_label"))
            try:
                label = label_path.read_text().strip()
            except OSError:
                label = ""
            temps.append((label, milli))
        if temps:
            devices.append(HwmonDevice(name=name, temps=tuple(temps)))
    return devices


def read_cpu_temp_c(hwmon_root: str = "/sys/class/hwmon") -> float | None:
    return select_cpu_temp(read_hwmon_devices(hwmon_root))


# ---------------------------------------------------------------------------
# GPU (nvidia-smi)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuStat:
    index: int
    name: str
    util_percent: float | None
    temp_c: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None


# Queried with numeric fields FIRST and the (possibly comma-containing) name
# LAST, so a GPU name with a comma can't shift the numeric columns.
_NVIDIA_SMI_QUERY = "utilization.gpu,temperature.gpu,memory.used,memory.total,name"


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for an unsupported field


def parse_nvidia_smi(text: str) -> list[GpuStat]:
    """Parse ``nvidia-smi`` CSV (one line per GPU) for the queried fields.

    Expects ``--query-gpu=<_NVIDIA_SMI_QUERY> --format=csv,noheader,nounits``:
    ``<util>, <temp>, <mem_used>, <mem_total>, <name>``. Lines with too few
    columns are skipped; an ``[N/A]`` numeric becomes None, not a crash.
    """
    gpus: list[GpuStat] = []
    for index, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        name = ", ".join(parts[4:]).strip() or "GPU"
        gpus.append(
            GpuStat(
                index=index,
                name=name,
                util_percent=_num(parts[0]),
                temp_c=_num(parts[1]),
                mem_used_mb=_num(parts[2]),
                mem_total_mb=_num(parts[3]),
            )
        )
    return gpus


def _run_nvidia_smi(timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        LOGGER.warning("nvidia-smi exited %s: %s", result.returncode, result.stderr.strip()[:200])
        return None
    return result.stdout


def read_gpus(query: Callable[[], str | None] = _run_nvidia_smi) -> list[GpuStat]:
    """Read every NVIDIA GPU's usage/temp/VRAM; empty list if none/unavailable."""
    output = query()
    if not output:
        return []
    return parse_nvidia_smi(output)


# ---------------------------------------------------------------------------
# Network counters (/proc/net/dev)
# ---------------------------------------------------------------------------

# Interface name prefixes that are virtual (loopback, containers, bridges,
# tunnels) — excluded so the throughput reflects real NIC traffic and container
# traffic isn't double-counted via its veth/bridge pair.
_VIRTUAL_PREFIXES = ("docker", "br-", "veth", "virbr", "vnet", "tap", "tun", "vmnet", "ifb")


def is_physical_iface(name: str) -> bool:
    name = name.strip()
    if name == "lo":
        return False
    return not any(name.startswith(prefix) for prefix in _VIRTUAL_PREFIXES)


def parse_net_dev(text: str) -> dict[str, tuple[int, int]]:
    """Parse ``/proc/net/dev`` into ``{iface: (rx_bytes, tx_bytes)}``.

    The first two lines are headers. Each data line is
    ``iface: rx_bytes rx_packets ... tx_bytes tx_packets ...`` — rx_bytes is the
    first counter, tx_bytes the ninth. Malformed lines are skipped.
    """
    result: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, rest = line.partition(":")
        fields = rest.split()
        if len(fields) < 9:
            continue
        try:
            rx_bytes = int(fields[0])
            tx_bytes = int(fields[8])
        except ValueError:
            continue
        result[name.strip()] = (rx_bytes, tx_bytes)
    return result


def total_bytes(counters: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Sum (rx, tx) byte counters across the physical interfaces only."""
    rx = tx = 0
    for name, (rx_bytes, tx_bytes) in counters.items():
        if not is_physical_iface(name):
            continue
        rx += rx_bytes
        tx += tx_bytes
    return rx, tx


def read_net_totals(proc_net_dev: str = "/proc/net/dev") -> tuple[int, int] | None:
    try:
        return total_bytes(parse_net_dev(Path(proc_net_dev).read_text()))
    except OSError:
        LOGGER.warning("Could not read %s", proc_net_dev)
        return None


# ---------------------------------------------------------------------------
# Olla load-balancer node health
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OllaNode:
    name: str
    status: str
    response_time: str
    success_rate: str
    model_count: int


@dataclass(frozen=True)
class OllaHealth:
    nodes: list[OllaNode]
    healthy_count: int
    total_count: int
    routable_count: int


def is_olla_url(base_url: str) -> bool:
    """True when the AI endpoint is an Olla proxy (path has an ``olla`` segment).

    Olla's Ollama-compatible path is ``…/olla/ollama`` (see config/.env); a plain
    Ollama is just ``host:11434`` with no such segment.
    """
    segments = [seg for seg in urlparse(base_url).path.split("/") if seg]
    return "olla" in segments


def olla_root(base_url: str) -> str:
    """The Olla management root (scheme://host:port) — its status API lives there,
    not under the ``/olla/<provider>`` proxy path used for model calls."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_olla_health(payload: object) -> OllaHealth | None:
    """Turn Olla's ``/internal/status/endpoints`` JSON into an :class:`OllaHealth`."""
    if not isinstance(payload, dict):
        return None  # a 200 with a non-object body (null/list/scalar) -> degrade
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list):
        return None
    nodes: list[OllaNode] = []
    for entry in endpoints:
        if not isinstance(entry, dict):
            continue
        nodes.append(
            OllaNode(
                name=str(entry.get("name", "?")),
                status=str(entry.get("status", "unknown")),
                response_time=str(entry.get("response_time", "")),
                success_rate=str(entry.get("success_rate", "")),
                model_count=_as_int(entry.get("model_count")),
            )
        )
    healthy = _as_int(
        payload.get("healthy_count"),
        sum(1 for node in nodes if node.status.lower() == "healthy"),
    )
    return OllaHealth(
        nodes=nodes,
        healthy_count=healthy,
        total_count=_as_int(payload.get("total_count"), len(nodes)),
        routable_count=_as_int(payload.get("routable_count")),
    )


def fetch_olla_health(
    base_url: str,
    *,
    timeout: float = 5.0,
    session: requests.Session | None = None,
) -> OllaHealth | None:
    """Fetch per-node health from the Olla balancer. None on any failure."""
    url = f"{olla_root(base_url)}/internal/status/endpoints"
    getter = session.get if session is not None else requests.get
    try:
        response = getter(url, timeout=timeout)
        response.raise_for_status()
        return parse_olla_health(response.json())
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        LOGGER.warning("Olla health fetch failed (%s)", url)
        return None


# ---------------------------------------------------------------------------
# Rolling-window sampling: a 1-minute average of every changing metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One moment's raw readings; reduced to averages by :func:`average_samples`."""

    ts: float
    cpu: CpuTimes | None
    cpu_temp_c: float | None
    gpus: tuple[GpuStat, ...]
    rx: int | None
    tx: int | None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None


@dataclass(frozen=True)
class MachineTelemetry:
    cpu_name: str | None = None
    cpu_threads: int | None = None
    cpu_percent: float | None = None
    cpu_temp_c: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    gpus: list[GpuStat] = field(default_factory=list)
    net_down_bps: float | None = None
    net_up_bps: float | None = None
    olla: OllaHealth | None = None


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _counter_rate(
    older: int | None, newer: int | None, dt: float
) -> float | None:
    """Per-second rate between two cumulative counters; None on reset/no-time."""
    if older is None or newer is None or dt <= 0:
        return None
    rate = (newer - older) / dt
    return rate if rate >= 0 else None  # negative => counter reset


def average_samples(samples: list[Sample], *, window_seconds: float) -> MachineTelemetry:
    """Reduce the trailing-window samples to 1-minute averages (pure).

    Counter metrics (CPU busy %, network bytes) are differenced from the oldest
    sample still inside the window to the newest; gauge metrics (CPU/GPU temp,
    GPU util) are the mean of the in-window readings; VRAM + names come from the
    newest sample. Identity (cpu_name/threads) and Olla are filled in by the
    caller. Empty/single-sample inputs yield None fields ("n/a"), never a crash.
    """
    if not samples:
        return MachineTelemetry()
    newest = samples[-1]
    target = newest.ts - window_seconds
    within = [s for s in samples if s.ts >= target] or [newest]
    oldest = within[0]
    has_span = oldest is not newest
    dt = newest.ts - oldest.ts

    cpu_percent = None
    if has_span and oldest.cpu is not None and newest.cpu is not None:
        cpu_percent = cpu_usage_percent(oldest.cpu, newest.cpu)

    cpu_temp = _mean([s.cpu_temp_c for s in within])
    mem_used = _mean([s.mem_used_mb for s in within])  # gauge -> mean over window
    mem_total = newest.mem_total_mb  # static -> latest

    down = _counter_rate(oldest.rx, newest.rx, dt) if has_span else None
    up = _counter_rate(oldest.tx, newest.tx, dt) if has_span else None

    gpus: list[GpuStat] = []
    for gpu in newest.gpus:
        utils = [sg.util_percent for s in within for sg in s.gpus if sg.index == gpu.index]
        temps = [sg.temp_c for s in within for sg in s.gpus if sg.index == gpu.index]
        gpus.append(
            replace(gpu, util_percent=_mean(utils), temp_c=_mean(temps))
        )

    return MachineTelemetry(
        cpu_percent=cpu_percent,
        cpu_temp_c=cpu_temp,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
        gpus=gpus,
        net_down_bps=down,
        net_up_bps=up,
    )


class MachineSampler:
    """Background thread sampling every changing metric on a rolling window.

    Sampling CPU jiffies, CPU temp, GPU stats, and network counters every
    ``sample_interval`` seconds lets :meth:`telemetry` return a trailing
    ``window_seconds`` average without ``/machine`` blocking to measure. The
    reader callables and ``now`` are injected so the windowing is testable
    without the clock, the filesystem, or a subprocess.
    """

    def __init__(
        self,
        stop_event: threading.Event,
        *,
        read_cpu: Callable[[], CpuTimes | None] = read_cpu_times,
        read_cpu_temp: Callable[[], float | None] = read_cpu_temp_c,
        read_mem: Callable[[], tuple[float, float] | None] = read_meminfo_mb,
        read_gpu: Callable[[], list[GpuStat]] = read_gpus,
        read_net: Callable[[], tuple[int, int] | None] = read_net_totals,
        now: Callable[[], float] = time.monotonic,
        sample_interval: float = 5.0,
        window_seconds: float = 60.0,
    ) -> None:
        self._stop = stop_event
        self._read_cpu = read_cpu
        self._read_cpu_temp = read_cpu_temp
        self._read_mem = read_mem
        self._read_gpu = read_gpu
        self._read_net = read_net
        self._now = now
        self._interval = sample_interval
        self._window = window_seconds
        self._lock = threading.Lock()
        self._samples: deque[Sample] = deque()

    def start(self) -> None:
        threading.Thread(target=self._run, name="machine-sampler", daemon=True).start()

    def _run(self) -> None:
        LOGGER.info("Machine sampler started (every %.0fs, %.0fs window)", self._interval, self._window)
        self.sample()  # prime immediately so the window starts filling at once
        while not self._stop.is_set():
            if self._stop.wait(self._interval):
                break
            try:
                self.sample()
            except Exception:
                LOGGER.exception("Machine sample failed")

    def sample(self) -> None:
        """Record one moment's readings and prune the stale tail."""
        net = self._read_net()
        mem = self._read_mem()
        sample = Sample(
            ts=self._now(),
            cpu=self._read_cpu(),
            cpu_temp_c=self._read_cpu_temp(),
            gpus=tuple(self._read_gpu()),
            rx=net[0] if net is not None else None,
            tx=net[1] if net is not None else None,
            mem_used_mb=mem[0] if mem is not None else None,
            mem_total_mb=mem[1] if mem is not None else None,
        )
        with self._lock:
            self._samples.append(sample)
            # Keep a little more than the window so there's always a point at or
            # before (now - window); drop everything older.
            cutoff = sample.ts - self._window - self._interval
            while len(self._samples) > 2 and self._samples[0].ts < cutoff:
                self._samples.popleft()

    def telemetry(self, window_seconds: float | None = None) -> MachineTelemetry:
        """Averages over the trailing window — pass a shorter ``window_seconds``
        for a more responsive "live" read, or None for the full configured window."""
        with self._lock:
            samples = list(self._samples)
        return average_samples(
            samples, window_seconds=window_seconds if window_seconds is not None else self._window
        )


# ---------------------------------------------------------------------------
# Assembly + formatting
# ---------------------------------------------------------------------------


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0f}%" if value is not None else "n/a"


def _fmt_temp(celsius: float | None) -> str:
    return f"{celsius:.0f}°C" if celsius is not None else "n/a"


def _fmt_gb(mb: float | None) -> str:
    return f"{mb / 1024:.1f}" if mb is not None else "n/a"


def human_rate(bytes_per_s: float | None) -> str:
    """A byte/s throughput as a compact human string (B/s, KB/s, MB/s, GB/s)."""
    if bytes_per_s is None:
        return "n/a"
    for unit, scale in (("GB/s", 1 << 30), ("MB/s", 1 << 20), ("KB/s", 1 << 10)):
        if bytes_per_s >= scale:
            return f"{bytes_per_s / scale:.1f} {unit}"
    return f"{bytes_per_s:.0f} B/s"


# The /machine command renders a LIVE message: an initial frame, then an edit
# every tick for the whole live span, settling into the full 1-minute average.
MACHINE_LIVE_SECONDS = 60.0
MACHINE_TICK_SECONDS = 1.0
# The "Now" numbers average over this short trailing window so they visibly move
# each second; the settled final frame uses the sampler's full window instead.
MACHINE_NOW_WINDOW = 5.0
# Olla health is point-in-time and a network round-trip, so it's refreshed at
# most this often during a live session rather than on every tick.
MACHINE_OLLA_REFRESH_SECONDS = 15.0


def _progress_bar(fraction: float, width: int = 16) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _present(value: str) -> bool:
    """True for a real reported value — Olla sends "N/A" for not-yet-measured fields."""
    return bool(value) and value.strip().lower() != "n/a"


def _metric_line(label: str, value: str, detail: str = "") -> str:
    """One aligned row: a left label, a right-aligned value column, then detail.

    The value column holds the percent (or a ↓/↑ arrow for the network rows) so
    every row's numbers line up under each other in the monospace block.
    """
    line = f"{label:<4} {value:>4}"
    return f"{line}   {detail}" if detail else line


def _mem_detail(used_mb: float | None, total_mb: float | None) -> str:
    return f"{_fmt_gb(used_mb)}/{_fmt_gb(total_mb)} GB"


def _number_lines(telemetry: MachineTelemetry) -> list[str]:
    """The live numbers as an aligned monospace block (rendered inside <pre>).

    One metric per row — CPU/GPU show usage% + temp, RAM/VRAM show usage% +
    used/total, and the network up/down rates get a row each.
    """
    lines = [_metric_line("CPU", _fmt_pct(telemetry.cpu_percent), _fmt_temp(telemetry.cpu_temp_c))]
    if telemetry.mem_total_mb is not None:
        lines.append(
            _metric_line(
                "RAM",
                _fmt_pct(_mem_percent(telemetry.mem_used_mb, telemetry.mem_total_mb)),
                _mem_detail(telemetry.mem_used_mb, telemetry.mem_total_mb),
            )
        )
    else:
        lines.append(_metric_line("RAM", "n/a"))
    if telemetry.gpus:
        for gpu in telemetry.gpus:
            lines.append(_metric_line("GPU", _fmt_pct(gpu.util_percent), _fmt_temp(gpu.temp_c)))
            if gpu.mem_total_mb is not None:
                lines.append(
                    _metric_line(
                        "VRAM",
                        _fmt_pct(_mem_percent(gpu.mem_used_mb, gpu.mem_total_mb)),
                        _mem_detail(gpu.mem_used_mb, gpu.mem_total_mb),
                    )
                )
    else:
        lines.append(_metric_line("GPU", "n/a"))
    lines.append(_metric_line("NET", "↓", human_rate(telemetry.net_down_bps)))
    lines.append(_metric_line("NET", "↑", human_rate(telemetry.net_up_bps)))
    return lines


def _spec_lines(telemetry: MachineTelemetry) -> list[str]:
    """The static hardware identity — shown separately from the live numbers."""
    specs: list[str] = []
    if telemetry.cpu_name:
        threads = f" · {telemetry.cpu_threads} threads" if telemetry.cpu_threads else ""
        specs.append(f"CPU: {telemetry.cpu_name}{threads}")
    if telemetry.mem_total_mb is not None:
        specs.append(f"RAM: {_fmt_gb(telemetry.mem_total_mb)} GB")
    for gpu in telemetry.gpus:
        vram = f" · {_fmt_gb(gpu.mem_total_mb)} GB" if gpu.mem_total_mb is not None else ""
        specs.append(f"GPU: {gpu.name}{vram}")
    return specs


def _olla_lines(health: OllaHealth) -> list[str]:
    lines = ["", f"🌐 **Olla cluster — {health.healthy_count}/{health.total_count} healthy**"]
    if not health.nodes:
        lines.append("  • no nodes reported")
        return lines
    for node in health.nodes:
        dot = "🟢" if node.status.lower() == "healthy" else "🔴"
        extras = []
        if _present(node.response_time):
            extras.append(node.response_time)
        if _present(node.success_rate):
            extras.append(f"{node.success_rate} ok")
        extras.append(f"{node.model_count} models")
        lines.append(f"  {dot} {node.name} — {' · '.join(extras)}")
    return lines


def format_machine(
    telemetry: MachineTelemetry,
    *,
    elapsed: float | None = None,
    total: float | None = None,
) -> str:
    """Render one /machine frame: a clean numbers block, then specs, then Olla.

    When ``elapsed``/``total`` are given (and the session is still running) the
    header shows a live progress bar and the numbers are the short-window "Now"
    read; otherwise it's the settled 1-minute average.
    """
    live = elapsed is not None and total is not None and elapsed < total
    lines: list[str] = []
    if live:
        bar = _progress_bar(elapsed / total if total else 1.0)
        lines.append("🖥️ **Machine** — live")
        lines.append(f"`[{bar}]` {int(elapsed)}s / {int(total)}s")
        lines.append("")
        lines.append("📊 **Now**")
    else:
        lines.append("🖥️ **Machine**")
        lines.append("")
        lines.append("📊 **Last minute (avg)**")
    # Numbers in a fenced block so the columns line up in a monospace box.
    lines.append("```")
    lines.extend(_number_lines(telemetry))
    lines.append("```")
    specs = _spec_lines(telemetry)
    if specs:
        lines.append("🧩 **Specs**")
        lines.extend(specs)
    if telemetry.olla is not None:
        lines.extend(_olla_lines(telemetry.olla))
    return "\n".join(lines)


# Back-compat alias: a single (non-live) snapshot of the settled averages.
def summarize_machine(telemetry: MachineTelemetry) -> str:
    return format_machine(telemetry)


class MachineFrames:
    """Renders /machine frames from a sampler, caching the static + slow parts.

    CPU identity is read once; Olla health (a network round-trip) is refreshed at
    most every ``olla_refresh_seconds`` so a 60-frame live session doesn't hammer
    the cluster. :meth:`frame` is the per-tick live renderer; :meth:`snapshot` is
    the one-shot (non-live) read used by the natural-language path.
    """

    def __init__(
        self,
        sampler: MachineSampler | None,
        *,
        ollama_base_url: str | None = None,
        ollama_enabled: bool = False,
        now_window: float = MACHINE_NOW_WINDOW,
        olla_refresh_seconds: float = MACHINE_OLLA_REFRESH_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sampler = sampler
        self._base_url = ollama_base_url
        self._wants_olla = bool(ollama_enabled and ollama_base_url and is_olla_url(ollama_base_url))
        self._now_window = now_window
        self._olla_refresh = olla_refresh_seconds
        self._now = now
        self._cpu_name, self._cpu_threads = read_cpu_model()
        self._lock = threading.Lock()
        self._olla: OllaHealth | None = None
        self._olla_at: float | None = None

    def _olla_health(self) -> OllaHealth | None:
        if not self._wants_olla:
            return None
        with self._lock:
            now = self._now()
            if self._olla_at is None or now - self._olla_at >= self._olla_refresh:
                self._olla = fetch_olla_health(self._base_url)  # type: ignore[arg-type]
                self._olla_at = now
            return self._olla

    def _telemetry(self, window_seconds: float | None) -> MachineTelemetry:
        telemetry = (
            self._sampler.telemetry(window_seconds=window_seconds)
            if self._sampler is not None
            else MachineTelemetry()
        )
        return replace(
            telemetry,
            cpu_name=self._cpu_name,
            cpu_threads=self._cpu_threads,
            olla=self._olla_health(),
        )

    def frame(self, elapsed: float, total: float) -> str:
        """One live frame; once ``elapsed >= total`` it's the settled 1-min avg."""
        live = elapsed < total
        telemetry = self._telemetry(self._now_window if live else None)
        return format_machine(telemetry, elapsed=elapsed, total=total)

    def snapshot(self) -> str:
        """A single, non-live 1-minute-average reply (natural-language path)."""
        return format_machine(self._telemetry(None))
