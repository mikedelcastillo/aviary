from __future__ import annotations

import threading

from lib.ai.intent import ACTIONS
from lib.telegram.commands import COMMAND_DESCRIPTIONS
from lib.machine import (
    CpuTimes,
    GpuStat,
    HwmonDevice,
    MachineFrames,
    MachineSampler,
    MachineTelemetry,
    OllaHealth,
    OllaNode,
    Sample,
    average_samples,
    cpu_usage_percent,
    fetch_olla_health,
    format_machine,
    human_rate,
    is_olla_url,
    is_physical_iface,
    olla_root,
    parse_cpu_model,
    parse_cpu_times,
    parse_meminfo,
    parse_net_dev,
    parse_nvidia_smi,
    parse_olla_health,
    read_gpus,
    select_cpu_temp,
    summarize_machine,
    total_bytes,
)


# --- CPU usage + identity --------------------------------------------------


def test_parse_cpu_times_totals_and_idle():
    t = parse_cpu_times("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 10 0 5 80 5 0 0 0 0 0\n")
    assert t is not None
    assert t.total == 1000  # sum of all fields
    assert t.idle == 850  # idle (800) + iowait (50)


def test_parse_cpu_times_malformed_is_none():
    assert parse_cpu_times("intr 1 2 3\nctxt 99\n") is None
    assert parse_cpu_times("cpu  10 20") is None  # too few columns


def test_cpu_usage_percent_basic():
    prev = CpuTimes(total=1000, idle=850)
    cur = CpuTimes(total=1100, idle=900)  # dt=100, idle delta=50 -> 50% busy
    assert cpu_usage_percent(prev, cur) == 50.0


def test_cpu_usage_percent_zero_or_negative_delta_is_none():
    same = CpuTimes(total=1000, idle=850)
    assert cpu_usage_percent(same, same) is None
    assert cpu_usage_percent(CpuTimes(1000, 900), CpuTimes(1100, 800)) is None  # idle wrap


def test_parse_cpu_model_name_and_threads():
    text = (
        "processor\t: 0\nmodel name\t: AMD Ryzen 7 7700\n"
        "processor\t: 1\nmodel name\t: AMD Ryzen 7 7700\n"
    )
    name, threads = parse_cpu_model(text)
    assert name == "AMD Ryzen 7 7700"
    assert threads == 2


def test_parse_cpu_model_missing_name_still_counts_threads():
    name, threads = parse_cpu_model("processor\t: 0\nBogoMIPS\t: 5\n")
    assert name is None
    assert threads == 1


def test_parse_cpu_model_empty_is_none():
    assert parse_cpu_model("") == (None, None)


def test_parse_meminfo_used_and_total():
    text = "MemTotal:       32000000 kB\nMemFree:         1000000 kB\nMemAvailable:   20000000 kB\n"
    used, total = parse_meminfo(text)
    assert total == 32000000 / 1024  # MemTotal
    assert used == (32000000 - 20000000) / 1024  # MemTotal - MemAvailable


def test_parse_meminfo_missing_available_is_none():
    assert parse_meminfo("MemTotal: 32000000 kB\nMemFree: 1000000 kB\n") is None


# --- CPU temperature -------------------------------------------------------


def test_select_cpu_temp_prefers_k10temp_tctl_over_bogus_acpitz():
    devices = [
        HwmonDevice("acpitz", (("", 16800),)),  # the bogus ~16 °C zone
        HwmonDevice("k10temp", (("Tctl", 77000), ("Tccd1", 76250))),
    ]
    assert select_cpu_temp(devices) == 77.0


def test_select_cpu_temp_prefers_package_label_on_intel():
    devices = [HwmonDevice("coretemp", (("Core 0", 60000), ("Package id 0", 65000)))]
    assert select_cpu_temp(devices) == 65.0


def test_select_cpu_temp_falls_back_to_first_temp_when_unlabeled():
    assert select_cpu_temp([HwmonDevice("coretemp", (("", 55000),))]) == 55.0


def test_select_cpu_temp_ignores_non_cpu_sensors():
    devices = [
        HwmonDevice("nvme", (("Composite", 40000),)),
        HwmonDevice("acpitz", (("", 16800),)),
    ]
    assert select_cpu_temp(devices) is None


# --- GPU (nvidia-smi) ------------------------------------------------------


def test_parse_nvidia_smi_single_gpu():
    gpus = parse_nvidia_smi("4, 45, 2276, 8151, NVIDIA GeForce RTX 5060")
    assert len(gpus) == 1
    gpu = gpus[0]
    assert (gpu.util_percent, gpu.temp_c, gpu.mem_used_mb, gpu.mem_total_mb) == (4.0, 45.0, 2276.0, 8151.0)
    assert gpu.name == "NVIDIA GeForce RTX 5060"


def test_parse_nvidia_smi_na_field_becomes_none():
    gpus = parse_nvidia_smi("[N/A], 50, 1000, 2000, Tesla T4")
    assert gpus[0].util_percent is None
    assert gpus[0].temp_c == 50.0


def test_parse_nvidia_smi_name_with_comma_keeps_numeric_columns():
    gpus = parse_nvidia_smi("5, 50, 100, 200, Weird, Edition GPU")
    assert gpus[0].util_percent == 5.0
    assert gpus[0].name == "Weird, Edition GPU"


def test_parse_nvidia_smi_skips_blank_and_short_lines():
    gpus = parse_nvidia_smi("\n4,45,1000\n0, 40, 500, 8000, GPU A\n")
    assert len(gpus) == 1
    assert gpus[0].name == "GPU A"


def test_read_gpus_empty_when_unavailable():
    assert read_gpus(query=lambda: None) == []
    assert read_gpus(query=lambda: "") == []


# --- Network counters ------------------------------------------------------


def test_is_physical_iface():
    assert is_physical_iface("eno1")
    assert is_physical_iface("wlp5s0")
    assert not is_physical_iface("lo")
    assert not is_physical_iface("docker0")
    assert not is_physical_iface("br-da17061a951a")
    assert not is_physical_iface("veth3f25a0")


def test_parse_net_dev_and_total_bytes_excludes_virtual():
    text = (
        "Inter-|   Receive                          |  Transmit\n"
        " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
        "    lo: 100 1 0 0 0 0 0 0 200 2 0 0 0 0 0 0\n"
        "  eno1: 1000 5 0 0 0 0 0 0 500 4 0 0 0 0 0 0\n"
        "docker0: 9 1 0 0 0 0 0 0 9 1 0 0 0 0 0 0\n"
    )
    counters = parse_net_dev(text)
    assert counters["eno1"] == (1000, 500)
    assert counters["lo"] == (100, 200)
    rx, tx = total_bytes(counters)  # only eno1 counts (lo + docker0 excluded)
    assert (rx, tx) == (1000, 500)


# --- Rolling-window averaging (average_samples) ----------------------------


def _s(ts, *, cpu=None, cpu_temp=None, gpus=(), rx=None, tx=None, mem_used=None, mem_total=None):
    return Sample(
        ts=ts,
        cpu=cpu,
        cpu_temp_c=cpu_temp,
        gpus=tuple(gpus),
        rx=rx,
        tx=tx,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
    )


def test_average_samples_cpu_usage_over_window():
    samples = [_s(0, cpu=CpuTimes(1000, 850)), _s(60, cpu=CpuTimes(1100, 900))]
    t = average_samples(samples, window_seconds=60)
    assert t.cpu_percent == 50.0  # 100 total / 50 idle delta -> 50% busy


def test_average_samples_cpu_temp_is_mean_of_readings():
    samples = [_s(0, cpu_temp=70.0), _s(30, cpu_temp=80.0)]
    assert average_samples(samples, window_seconds=60).cpu_temp_c == 75.0


def test_average_samples_ram_used_meaned_total_latest():
    samples = [_s(0, mem_used=8000.0, mem_total=32000.0), _s(20, mem_used=10000.0, mem_total=32000.0)]
    t = average_samples(samples, window_seconds=60)
    assert t.mem_used_mb == 9000.0  # gauge -> mean
    assert t.mem_total_mb == 32000.0  # static -> latest


def test_average_samples_network_rate():
    samples = [_s(0, rx=0, tx=0), _s(60, rx=60 * 1024, tx=30 * 1024)]
    t = average_samples(samples, window_seconds=60)
    assert round(t.net_down_bps) == 1024  # 60 KiB over 60 s
    assert round(t.net_up_bps) == 512


def test_average_samples_gpu_util_temp_meaned_vram_latest():
    g_old = GpuStat(0, "RTX 5060", 0.0, 40.0, 2000.0, 8000.0)
    g_new = GpuStat(0, "RTX 5060", 8.0, 50.0, 2200.0, 8000.0)
    t = average_samples([_s(0, gpus=[g_old]), _s(20, gpus=[g_new])], window_seconds=60)
    assert len(t.gpus) == 1
    assert t.gpus[0].util_percent == 4.0  # (0 + 8) / 2
    assert t.gpus[0].temp_c == 45.0  # (40 + 50) / 2
    assert t.gpus[0].mem_used_mb == 2200.0  # latest, not averaged
    assert t.gpus[0].name == "RTX 5060"


def test_average_samples_empty_is_all_none():
    t = average_samples([], window_seconds=60)
    assert t.cpu_percent is None
    assert t.cpu_temp_c is None
    assert t.gpus == []
    assert t.net_down_bps is None and t.net_up_bps is None


def test_average_samples_single_sample_has_no_counter_rates():
    samples = [_s(10, cpu=CpuTimes(1000, 850), cpu_temp=70.0, rx=5, tx=5)]
    t = average_samples(samples, window_seconds=60)
    assert t.cpu_percent is None  # counters need two points
    assert t.net_down_bps is None
    assert t.cpu_temp_c == 70.0  # but a gauge reading is available


def test_average_samples_uses_recent_window_not_ancient_idle():
    # Idle for ages, then a burst in the last 10 s: the rate reflects the burst.
    samples = [_s(0, rx=0, tx=0), _s(150, rx=0, tx=0), _s(160, rx=6000, tx=3000)]
    t = average_samples(samples, window_seconds=60)
    assert round(t.net_down_bps) == 600  # 6000 B over the 10 s within the window
    assert round(t.net_up_bps) == 300


def test_average_samples_counter_reset_is_none():
    samples = [_s(0, rx=100_000, tx=100_000), _s(30, rx=50_000, tx=40_000)]
    t = average_samples(samples, window_seconds=60)
    assert t.net_down_bps is None and t.net_up_bps is None


# --- MachineSampler (threaded, with injected readers) ----------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _reader(values):
    box = {"i": 0}

    def read():
        i = min(box["i"], len(values) - 1)
        box["i"] += 1
        return values[i]

    return read


def test_machine_sampler_collects_and_averages_all_metrics():
    clock = _Clock()
    sampler = MachineSampler(
        threading.Event(),
        read_cpu=_reader([CpuTimes(1000, 850), CpuTimes(1100, 900)]),
        read_cpu_temp=_reader([70.0, 80.0]),
        read_gpu=_reader(
            [
                [GpuStat(0, "G", 0.0, 40.0, 2000.0, 8000.0)],
                [GpuStat(0, "G", 8.0, 50.0, 2200.0, 8000.0)],
            ]
        ),
        read_net=_reader([(0, 0), (60 * 1024, 30 * 1024)]),
        now=clock,
        sample_interval=10.0,
        window_seconds=60.0,
    )
    clock.t = 0.0
    sampler.sample()
    clock.t = 60.0
    sampler.sample()
    t = sampler.telemetry()
    assert t.cpu_percent == 50.0
    assert t.cpu_temp_c == 75.0
    assert round(t.net_down_bps) == 1024
    assert t.gpus[0].util_percent == 4.0
    assert t.gpus[0].temp_c == 45.0


def test_machine_sampler_telemetry_before_two_samples():
    sampler = MachineSampler(
        threading.Event(),
        read_cpu=_reader([CpuTimes(1, 1)]),
        read_cpu_temp=_reader([60.0]),
        read_gpu=_reader([[]]),
        read_net=_reader([(0, 0)]),
        now=_Clock(),
    )
    sampler.sample()
    t = sampler.telemetry()
    assert t.cpu_percent is None  # warming up
    assert t.cpu_temp_c == 60.0


# --- Olla node health ------------------------------------------------------


def test_is_olla_url():
    assert is_olla_url("http://localhost:40114/olla/ollama")
    assert not is_olla_url("http://localhost:11434")
    assert not is_olla_url("http://host:11434/api")


def test_olla_root_strips_proxy_path():
    assert olla_root("http://localhost:40114/olla/ollama") == "http://localhost:40114"


def test_parse_olla_health_extracts_nodes_and_counts():
    payload = {
        "endpoints": [
            {"name": "localhost", "status": "healthy", "response_time": "0ms", "success_rate": "10.4%", "model_count": 2},
            {"name": "192.168.1.2", "status": "offline", "response_time": "5ms", "success_rate": "88.9%", "model_count": 18},
        ],
        "total_count": 2,
        "healthy_count": 1,
        "routable_count": 1,
    }
    health = parse_olla_health(payload)
    assert health is not None
    assert (health.total_count, health.healthy_count, health.routable_count) == (2, 1, 1)
    assert health.nodes[0].name == "localhost"
    assert health.nodes[0].model_count == 2
    assert health.nodes[1].status == "offline"


def test_parse_olla_health_infers_counts_when_absent():
    payload = {"endpoints": [{"name": "a", "status": "healthy"}, {"name": "b", "status": "down"}]}
    health = parse_olla_health(payload)
    assert health.healthy_count == 1  # inferred from node statuses
    assert health.total_count == 2
    assert health.nodes[0].model_count == 0  # missing -> 0, not a crash


def test_parse_olla_health_missing_endpoints_is_none():
    assert parse_olla_health({"proxy": {}}) is None


def test_parse_olla_health_non_object_payload_is_none():
    # A 200 with a non-object JSON body must degrade, not raise AttributeError.
    assert parse_olla_health([{"name": "a"}]) is None
    assert parse_olla_health(None) is None
    assert parse_olla_health("unauthorized") is None


def test_fetch_olla_health_uses_status_endpoints_route():
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"endpoints": [{"name": "a", "status": "healthy"}], "total_count": 1, "healthy_count": 1}

    class FakeSession:
        def get(self, url, timeout):
            captured["url"] = url
            return FakeResponse()

    health = fetch_olla_health("http://localhost:40114/olla/ollama", session=FakeSession())
    assert captured["url"] == "http://localhost:40114/internal/status/endpoints"
    assert health is not None and health.healthy_count == 1


def test_fetch_olla_health_degrades_on_non_object_json():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return ["not", "an", "object"]  # valid JSON, wrong shape

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    assert fetch_olla_health("http://h:40114/olla/ollama", session=FakeSession()) is None


# --- Formatting ------------------------------------------------------------


def test_human_rate_units():
    assert human_rate(None) == "n/a"
    assert human_rate(0) == "0 B/s"
    assert human_rate(512) == "512 B/s"
    assert human_rate(1 << 10) == "1.0 KB/s"
    assert human_rate(1 << 20) == "1.0 MB/s"
    assert human_rate(1 << 30) == "1.0 GB/s"


def _full_telemetry():
    return MachineTelemetry(
        cpu_name="AMD Ryzen 7 7700",
        cpu_threads=16,
        cpu_percent=23.4,
        cpu_temp_c=77.0,
        mem_used_mb=16384.0,
        mem_total_mb=32768.0,
        gpus=[GpuStat(0, "NVIDIA GeForce RTX 5060", 4.0, 45.0, 2276.0, 8151.0)],
        net_down_bps=1.5 * (1 << 20),
        net_up_bps=300 * (1 << 10),
        olla=OllaHealth(
            nodes=[OllaNode("localhost", "healthy", "0ms", "10.4%", 2)],
            healthy_count=1,
            total_count=1,
            routable_count=1,
        ),
    )


def test_format_machine_snapshot_separates_numbers_and_specs():
    text = format_machine(_full_telemetry())
    assert "🖥️ **Machine**" in text and "— live" not in text
    assert "📊 **Last minute (avg)**" in text
    # numbers — the moving values, in a monospace block
    assert "```" in text
    assert "23%" in text and "77°C" in text  # CPU
    assert "RAM" in text and "50%" in text and "16.0/32.0 GB" in text  # RAM in the mix
    assert "4%" in text and "45°C" in text  # GPU usage + temp (VRAM moved off this line)
    assert "VRAM" in text and "2.2/8.0 GB" in text  # VRAM its own line, with %
    # net up/down on separate rows
    assert "↓" in text and "↑" in text
    assert "1.5 MB/s" in text and "300.0 KB/s" in text
    # specs — the static hardware identity, kept separate
    assert "🧩 **Specs**" in text
    assert "CPU: AMD Ryzen 7 7700 · 16 threads" in text
    assert "RAM: 32.0 GB" in text
    assert "GPU: NVIDIA GeForce RTX 5060 · 8.0 GB" in text
    # olla cluster
    assert "🌐 **Olla cluster — 1/1 healthy**" in text
    assert "🟢 localhost" in text and "2 models" in text


def test_number_block_splits_vram_and_net_into_their_own_rows():
    text = format_machine(
        MachineTelemetry(
            cpu_percent=10.0,
            cpu_temp_c=50.0,
            mem_used_mb=8000.0,
            mem_total_mb=16000.0,
            gpus=[GpuStat(0, "G", 20.0, 60.0, 2000.0, 8000.0)],
            net_down_bps=1 << 20,
            net_up_bps=1 << 19,
        )
    )
    rows = text.splitlines()
    assert sum(1 for r in rows if r.startswith("VRAM")) == 1  # VRAM its own row
    assert sum(1 for r in rows if r.startswith("NET")) == 2  # ↓ and ↑ on separate rows
    gpu_row = next(r for r in rows if r.startswith("GPU ") and "%" in r)
    assert "GB" not in gpu_row  # VRAM no longer rides on the GPU row
    assert "25%" in text  # VRAM 2000/8000 -> 25%


def test_format_machine_live_has_progress_bar_and_now_label():
    text = format_machine(
        MachineTelemetry(cpu_name="CPU X", cpu_threads=8, cpu_percent=10.0, cpu_temp_c=50.0),
        elapsed=15.0,
        total=60.0,
    )
    assert "🖥️ **Machine** — live" in text
    assert "15s / 60s" in text
    assert "📊 **Now**" in text
    assert "█" in text and "░" in text  # progress bar drawn
    assert "🧩 **Specs**" in text  # specs still shown live


def test_format_machine_final_frame_settles_to_average():
    # elapsed >= total -> no longer live -> the settled summary, no progress bar.
    text = format_machine(MachineTelemetry(cpu_percent=10.0, cpu_temp_c=50.0), elapsed=60.0, total=60.0)
    assert "— live" not in text
    assert "📊 **Last minute (avg)**" in text


def test_summarize_machine_alias_is_non_live_format():
    telemetry = MachineTelemetry(cpu_percent=5.0, cpu_temp_c=40.0)
    assert summarize_machine(telemetry) == format_machine(telemetry)


def test_format_machine_degrades_to_na():
    text = format_machine(MachineTelemetry())
    assert "n/a" in text
    assert "🧩" not in text  # no specs section when nothing is known
    assert "🌐" not in text  # no cluster section when not an Olla endpoint


def test_format_machine_marks_unhealthy_node():
    text = format_machine(
        MachineTelemetry(
            cpu_percent=10.0,
            cpu_temp_c=50.0,
            olla=OllaHealth(
                nodes=[OllaNode("192.168.1.2", "offline", "", "", 0)],
                healthy_count=0,
                total_count=1,
                routable_count=0,
            ),
        )
    )
    assert "🔴 192.168.1.2" in text
    assert "0/1 healthy" in text


# --- MachineFrames (live frame renderer) -----------------------------------


class _FakeSampler:
    def __init__(self, telemetry):
        self._telemetry = telemetry
        self.window = "unset"

    def telemetry(self, window_seconds=None):
        self.window = window_seconds
        return self._telemetry


def test_machine_frames_live_uses_short_window():
    fake = _FakeSampler(MachineTelemetry(cpu_percent=50.0, cpu_temp_c=70.0))
    frames = MachineFrames(fake, ollama_enabled=False, now_window=5.0)
    text = frames.frame(10.0, 60.0)
    assert fake.window == 5.0  # live -> responsive short window
    assert "— live" in text and "📊 **Now**" in text
    assert "50%" in text and "70°C" in text


def test_machine_frames_final_and_snapshot_use_full_window():
    fake = _FakeSampler(MachineTelemetry(cpu_percent=12.0, cpu_temp_c=55.0))
    frames = MachineFrames(fake, ollama_enabled=False)
    frames.frame(60.0, 60.0)
    assert fake.window is None  # settled -> full window
    frames.snapshot()
    assert fake.window is None  # snapshot -> full window
    assert "🌐" not in frames.snapshot()  # olla disabled -> no cluster section


def test_machine_frames_caches_olla_health(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, **kwargs):
        calls["n"] += 1
        return OllaHealth([OllaNode("a", "healthy", "1ms", "100%", 2)], 1, 1, 1)

    monkeypatch.setattr("lib.machine.fetch_olla_health", fake_fetch)
    clock = _Clock()
    frames = MachineFrames(
        _FakeSampler(MachineTelemetry()),
        ollama_base_url="http://h:40114/olla/ollama",
        ollama_enabled=True,
        olla_refresh_seconds=15.0,
        now=clock,
    )
    clock.t = 0.0
    text = frames.frame(0.0, 60.0)
    clock.t = 5.0
    frames.frame(5.0, 60.0)  # within the refresh window -> served from cache
    assert calls["n"] == 1
    clock.t = 20.0
    frames.frame(20.0, 60.0)  # past the refresh window -> refetched
    assert calls["n"] == 2
    assert "🌐 **Olla cluster — 1/1 healthy**" in text


def test_machine_sampler_telemetry_respects_window_override():
    clock = _Clock()
    sampler = MachineSampler(
        threading.Event(),
        read_cpu=_reader([CpuTimes(0, 0), CpuTimes(100, 50), CpuTimes(200, 100)]),
        read_cpu_temp=_reader([10.0, 20.0, 30.0]),
        read_gpu=_reader([[], [], []]),
        read_net=_reader([(0, 0), (1000, 500), (2000, 1000)]),
        now=clock,
        sample_interval=10.0,
        window_seconds=60.0,
    )
    for t in (0.0, 10.0, 20.0):
        clock.t = t
        sampler.sample()
    assert sampler.telemetry().cpu_temp_c == 20.0  # full window -> mean(10,20,30)
    assert sampler.telemetry(window_seconds=5.0).cpu_temp_c == 30.0  # short -> newest only


# --- Wiring ----------------------------------------------------------------


def test_machine_command_and_action_registered():
    assert "/machine" in COMMAND_DESCRIPTIONS
    assert "machine" in ACTIONS
