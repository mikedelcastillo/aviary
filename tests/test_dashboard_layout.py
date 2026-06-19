"""Unit tests for the content-aware camera-band layout math.

The mode/column decisions are factored into pure helper functions that take only
(width, height, n), so we can assert the layout behaviour without standing up a
real terminal. Importing lib.dashboard pulls ultralytics in transitively (via
lib.stats -> lib.detector); that's expected — mirrors tests/test_telegram_commands.py.
"""

from __future__ import annotations

import math
import threading

from lib.dashboard import (
    COMPACT_MIN_CELL_WIDTH,
    DETAIL_MIN_CELL_WIDTH,
    DETAIL_PANEL_HEIGHT,
    Dashboard,
    _camera_layout_mode,
    _compact_columns,
    _compact_rows,
    _detail_columns,
    _discovery_columns,
)
from lib.discovery import HOST_FAILED, HOST_FOUND, HOST_TESTING, DiscoveryProgress
from lib.objects import ObjectRegistry
from lib.stats import CameraStats


def _make_stats(count: int) -> dict[str, CameraStats]:
    registry = ObjectRegistry()
    stats: dict[str, CameraStats] = {}
    for i in range(count):
        name = f"camera-10.0.0.{i + 1}"
        stats[name] = CameraStats(name, 0.25, registry)
    return stats


# -- mode decision ----------------------------------------------------------


def test_detailed_when_few_cameras_and_tall_band() -> None:
    # 2 cameras, wide enough for 2 columns -> 1 row -> fits a tall band.
    assert _camera_layout_mode(width=120, band_height=40, n=2) == "detailed"


def test_compact_when_band_too_short_for_grid() -> None:
    # Many cameras with only enough height for a fraction of the grid rows.
    assert _camera_layout_mode(width=120, band_height=DETAIL_PANEL_HEIGHT, n=12) == "compact"


def test_compact_when_many_cameras_cannot_fit_detailed() -> None:
    # Even a wide terminal can't fit 30 detailed panels in a 20-line band.
    assert _camera_layout_mode(width=200, band_height=20, n=30) == "compact"


def test_zero_cameras_reports_compact() -> None:
    assert _camera_layout_mode(width=80, band_height=10, n=0) == "compact"


def test_mode_boundary_exact_fit_is_detailed() -> None:
    # Width 120 // 28 = 4 columns, capped at n=4 -> 1 row; band exactly one panel
    # tall must still count as a fit (uses <=, not <).
    assert _detail_columns(120, 4) == 4
    assert _camera_layout_mode(width=120, band_height=DETAIL_PANEL_HEIGHT, n=4) == "detailed"


# -- detailed columns -------------------------------------------------------


def test_detail_single_column_only_when_one_camera() -> None:
    assert _detail_columns(width=200, n=1) == 1


def test_detail_at_least_two_columns_when_n_ge_2() -> None:
    # A very narrow terminal would compute 1 column by width, but the invariant
    # forces at least 2 columns whenever there are >= 2 cameras.
    assert _detail_columns(width=DETAIL_MIN_CELL_WIDTH, n=5) == 2


def test_detail_columns_never_exceed_camera_count() -> None:
    assert _detail_columns(width=1000, n=3) == 3


def test_detail_columns_scale_with_width() -> None:
    # 140 // 28 = 5, and there are plenty of cameras to fill them.
    assert _detail_columns(width=140, n=10) == 5


# -- compact columns/rows ---------------------------------------------------


def test_compact_columns_at_least_one() -> None:
    assert _compact_columns(width=10, n=4) == 1


def test_compact_columns_scale_with_width() -> None:
    # 130 // 26 = 5 columns available, 8 cameras -> 5 columns.
    assert _compact_columns(width=130, n=8) == 5


def test_compact_columns_never_exceed_camera_count() -> None:
    assert _compact_columns(width=1000, n=3) == 3


def test_compact_rows_cover_every_camera() -> None:
    # Whatever the packing, rows * columns must be able to hold all n cameras so
    # nothing is dropped from the dense view.
    for n in (1, 2, 5, 12, 33):
        for width in (30, 80, 200):
            columns = _compact_columns(width, n)
            rows = _compact_rows(n, columns)
            assert rows * columns >= n


# -- end-to-end band invariants (no real terminal needed) -------------------


def test_compact_band_shows_every_camera_status() -> None:
    stats = _make_stats(12)
    lock = threading.Lock()
    dash = Dashboard(stats, ObjectRegistry(), stats_lock=lock)
    snaps = dash._camera_snapshots()
    # Force compact by giving a short band.
    panel, height = dash._compact_band(snaps, width=120, max_band=20)
    # Render to plain text and assert each camera's shortened host appears.
    plain = dash.console.render_str  # noqa: F841 (kept for clarity of intent)
    from rich.console import Console

    text = "\n".join(
        seg.text for line in Console(width=120).render_lines(panel) for seg in line
    )
    for i in range(12):
        assert f"10.0.0.{i + 1}" in text


def test_band_uses_lock_snapshot_safely() -> None:
    # The snapshot helper must copy under the lock; calling it should never raise
    # even though it touches the live dict.
    stats = _make_stats(3)
    dash = Dashboard(stats, ObjectRegistry(), stats_lock=threading.Lock())
    snaps = dash._camera_snapshots()
    assert len(snaps) == 3


def test_zero_camera_band_is_placeholder() -> None:
    dash = Dashboard({}, ObjectRegistry(), stats_lock=threading.Lock())
    renderable, height = dash._camera_band([], width=80, max_band=20)
    from rich.console import Console

    text = "\n".join(
        seg.text for line in Console(width=80).render_lines(renderable) for seg in line
    )
    assert "/discover" in text


def test_dashboard_creates_lock_when_none_supplied() -> None:
    dash = Dashboard(_make_stats(2), ObjectRegistry(), stats_lock=None)
    assert isinstance(dash._stats_lock, type(threading.Lock()))


def test_detailed_band_two_columns_for_two_cameras() -> None:
    stats = _make_stats(2)
    dash = Dashboard(stats, ObjectRegistry(), stats_lock=threading.Lock())
    snaps = dash._camera_snapshots()
    grid, height = dash._detailed_band(snaps, width=120, max_band=40)
    # 2 cameras -> 2 columns -> 1 row -> one panel tall.
    assert math.ceil(2 / _detail_columns(120, 2)) == 1
    assert height == DETAIL_PANEL_HEIGHT


# -- discovery debug grid ---------------------------------------------------


def test_discovery_columns_fill_width() -> None:
    # 80 // DISCOVERY_CELL_WIDTH(4) = 20 columns; plenty of hosts to fill them.
    assert _discovery_columns(width=80, n=254) == 20


def test_discovery_columns_never_exceed_host_count() -> None:
    assert _discovery_columns(width=1000, n=3) == 3


def test_discovery_columns_at_least_one() -> None:
    assert _discovery_columns(width=2, n=10) == 1


def _progress_for(hosts: list[str]) -> DiscoveryProgress:
    progress = DiscoveryProgress()
    progress.begin(hosts, "192.168.1.")
    return progress


def test_discovery_band_shows_every_octet_and_states() -> None:
    hosts = [f"192.168.1.{i}" for i in range(1, 21)]
    progress = _progress_for(hosts)
    progress.mark("192.168.1.5", HOST_FOUND)
    progress.mark("192.168.1.6", HOST_TESTING)
    progress.mark("192.168.1.7", HOST_FAILED)

    dash = Dashboard({}, ObjectRegistry(), stats_lock=threading.Lock())
    snap = progress.snapshot()
    panel, height = dash._discovery_band(snap, width=100, max_band=30)

    from rich.console import Console

    console = Console(width=100)
    text = "\n".join(
        seg.text
        for line in console.render_lines(panel, console.options.update(height=height))
        for seg in line
    )
    # Every scanned host's last octet is present (nothing hidden)...
    for i in range(1, 21):
        assert f"{i:>2}" in text
    # ...and the title surfaces the scope + a running found count.
    assert "discovery — scanning 192.168.1.0/24" in text
    assert "1 found" in text


def test_render_switches_to_discovery_grid_when_active() -> None:
    hosts = [f"192.168.1.{i}" for i in range(1, 6)]
    progress = _progress_for(hosts)  # active until end() is called
    dash = Dashboard(
        {}, ObjectRegistry(), stats_lock=threading.Lock(), discovery_progress=progress
    )
    layout = dash._render()

    from rich.console import Console

    text = "\n".join(
        seg.text
        for line in Console(width=100, height=40).render_lines(layout)
        for seg in line
    )
    assert "discovery — scanning" in text
    # Once the sweep ends, the band reverts to the camera placeholder.
    progress.end()
    layout = dash._render()
    text = "\n".join(
        seg.text
        for line in Console(width=100, height=40).render_lines(layout)
        for seg in line
    )
    assert "discovery — scanning" not in text
    assert "/discover" in text
