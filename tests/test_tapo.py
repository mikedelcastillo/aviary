from __future__ import annotations

from lib.tapo import (
    REBOOT_COOLDOWN_SECONDS,
    FindFlash,
    RebootGuard,
    TapoControl,
    match_camera_hosts,
)


class FakeCamera:
    """A pytapo stand-in modelling the REAL lamp protocol (verified live):
    ``getWhitelampStatus`` reports a burn via ``rest_time > 0`` and
    ``reverseWhitelampStatus`` toggles it. ``whitelamp_calls`` records the
    lamp state AFTER each accepted toggle. Failures injectable per API."""

    def __init__(
        self,
        *,
        whitelamp_ok: bool = True,
        force_ok: bool = True,
        floodlight_ok: bool = True,
        reboot_ok: bool = True,
    ) -> None:
        self.whitelamp_ok = whitelamp_ok
        self.force_ok = force_ok
        self.floodlight_ok = floodlight_ok
        self.reboot_ok = reboot_ok
        self.burning = False
        self.whitelamp_calls: list[bool] = []
        self.force_calls: list[bool] = []
        self.floodlight_calls: list[bool] = []
        self.reboots = 0

    def getWhitelampStatus(self) -> dict:
        if not self.whitelamp_ok:
            raise RuntimeError("whitelamp unsupported")
        return {"status": 1 if self.burning else 0, "rest_time": 1800 if self.burning else 0}

    def reverseWhitelampStatus(self) -> None:
        if not self.whitelamp_ok:
            raise RuntimeError("whitelamp unsupported")
        self.burning = not self.burning
        self.whitelamp_calls.append(self.burning)

    def setForceWhitelampState(self, enable: bool) -> None:
        if not self.force_ok:
            raise RuntimeError("force flag unsupported")
        self.force_calls.append(enable)

    def getForceWhitelampState(self) -> bool:
        if not self.force_ok:
            raise RuntimeError("force flag unsupported")
        return bool(self.force_calls and self.force_calls[-1])

    def manualFloodlightOp(self, status: bool) -> None:
        if not self.floodlight_ok:
            raise RuntimeError("floodlight unsupported")
        self.floodlight_calls.append(status)

    def reboot(self) -> None:
        if not self.reboot_ok:
            raise RuntimeError("reboot refused")
        self.reboots += 1


def control(cameras: dict[str, FakeCamera], clock=lambda: 0.0, **kwargs) -> TapoControl:
    def connect(host: str, password: str):
        camera = cameras.get(host)
        if camera is None:
            raise ConnectionError("unreachable")
        return camera

    return TapoControl(
        "cloud-secret", connect=connect, clock=clock, release_delay=0.0, **kwargs
    )


# -- enabled / dormant --------------------------------------------------------


def test_disabled_without_cloud_password() -> None:
    ctl = TapoControl("", connect=lambda h, p: FakeCamera())
    assert ctl.enabled is False
    assert ctl.manual_set("192.168.1.19", True) is False
    assert ctl.reboot("192.168.1.19") is False


def test_enabled_with_cloud_password() -> None:
    assert control({}).enabled is True


# -- flash cascade -------------------------------------------------------------


def test_manual_set_uses_whitelamp_burn_first() -> None:
    camera = FakeCamera()
    ctl = control({"h": camera})
    assert ctl.manual_set("h", True) is True
    assert camera.whitelamp_calls == [True]  # one toggle, now burning
    assert camera.force_calls == [] and camera.floodlight_calls == []
    assert ctl.is_on("h") is True


def test_manual_set_skips_the_toggle_when_already_burning() -> None:
    # reverseWhitelampStatus is a TOGGLE — sending it to an already-burning
    # lamp would turn it OFF. The state is read first.
    camera = FakeCamera()
    camera.burning = True  # e.g. lit from the Tapo app
    ctl = control({"h": camera})
    assert ctl.manual_set("h", True) is True
    assert camera.whitelamp_calls == []  # no toggle sent


def test_manual_set_falls_back_to_force_flag_then_floodlight() -> None:
    # Firmware without the manual burn API: the force flag is tried next.
    camera = FakeCamera(whitelamp_ok=False)
    ctl = control({"h": camera})
    assert ctl.manual_set("h", True) is True
    assert camera.force_calls == [True] and camera.floodlight_calls == []

    # And without the force flag either, the floodlight op is the last resort.
    camera2 = FakeCamera(whitelamp_ok=False, force_ok=False)
    ctl2 = control({"h": camera2})
    assert ctl2.manual_set("h", True) is True
    assert camera2.floodlight_calls == [True]


def test_manual_set_reports_failure_when_no_lamp() -> None:
    camera = FakeCamera(whitelamp_ok=False, force_ok=False, floodlight_ok=False)
    ctl = control({"h": camera})
    assert ctl.manual_set("h", True) is False
    assert ctl.is_on("h") is False


def test_set_lamp_fails_honestly_when_the_toggle_does_not_take() -> None:
    # The toggle is read back; a lamp that stays in the wrong state is a
    # failure, not a silent success.
    class StuckCamera(FakeCamera):
        def reverseWhitelampStatus(self) -> None:
            pass  # accepts the call but the lamp never changes

    ctl = control({"h": StuckCamera()})
    assert ctl.manual_set("h", True) is False


def test_manual_set_unreachable_camera_fails_quietly() -> None:
    ctl = control({})  # connect raises for every host
    assert ctl.manual_set("gone", True) is False


# -- find/manual ownership ------------------------------------------------------


def test_find_lights_only_unowned_hosts_and_restores_them() -> None:
    a, b = FakeCamera(), FakeCamera()
    ctl = control({"a": a, "b": b})
    ctl.manual_set("b", True)  # the user already forced b on

    token = object()
    lit = ctl.find_start(["a", "b"], token)
    assert lit == ["a"]  # b is manually owned, never adopted

    ctl.find_stop(lit, token)
    assert a.whitelamp_calls == [True, False]  # lit then restored
    assert b.whitelamp_calls == [True]  # untouched by the search
    assert ctl.is_on("b") is True


def test_manual_off_releases_a_find_lamp_and_find_stop_skips_it() -> None:
    camera = FakeCamera()
    ctl = control({"a": camera})
    token = object()
    lit = ctl.find_start(["a"], token)
    assert lit == ["a"]
    # The user says off mid-search; their word is final.
    assert ctl.manual_set("a", False) is True
    ctl.find_stop(lit, token)
    # on (find), off (manual) — and no second off from find_stop.
    assert camera.whitelamp_calls == [True, False]


def test_find_stop_keeps_lamp_marked_lit_when_off_fails() -> None:
    camera = FakeCamera()
    ctl = control({"a": camera})
    token = object()
    lit = ctl.find_start(["a"], token)
    camera.whitelamp_ok = False
    camera.force_ok = False
    camera.floodlight_ok = False
    ctl.find_stop(lit, token)
    # The lamp may physically still be on: stay marked lit so a later
    # /flash off (or toggle) can still address it.
    assert ctl.is_on("a") is True


def test_find_start_skips_already_lit_hosts() -> None:
    camera = FakeCamera()
    ctl = control({"a": camera})
    ctl.manual_set("a", True)
    ctl.manual_set("a", False)
    ctl.manual_set("a", True)
    assert ctl.find_start(["a"], object()) == []


def test_replacement_search_shares_lamps_instead_of_being_doused() -> None:
    # A replaced /find's teardown overlaps the new search's start. The new
    # search must JOIN the lit lamp's owners, and the old teardown must leave
    # it burning — otherwise the replacement runs its whole search in the dark.
    camera = FakeCamera()
    ctl = control({"a": camera})
    token_a, token_b = object(), object()
    lit_a = ctl.find_start(["a"], token_a)  # search A lights the lamp
    lit_b = ctl.find_start(["a"], token_b)  # replacement B joins it
    assert lit_a == ["a"] and lit_b == ["a"]

    ctl.find_stop(lit_a, token_a)  # A's late teardown: B still owns the lamp
    assert ctl.is_on("a") is True
    assert camera.whitelamp_calls == [True]  # never turned off under B

    ctl.find_stop(lit_b, token_b)  # B ends: the last owner restores the lamp
    assert ctl.is_on("a") is False
    assert camera.whitelamp_calls == [True, False]


def test_reverse_ordered_replacement_cannot_douse_the_new_search() -> None:
    # The nastier interleaving: the NEW search B lights first; the cancelled
    # old search A's late start then joins and immediately tears down. The
    # lamp must survive for B — this is the direction single-owner adoption
    # got wrong.
    camera = FakeCamera()
    ctl = control({"a": camera})
    token_old, token_new = object(), object()
    lit_new = ctl.find_start(["a"], token_new)  # replacement B lights first
    lit_old = ctl.find_start(["a"], token_old)  # stale A joins late...
    ctl.find_stop(lit_old, token_old)  # ...and tears down immediately
    assert ctl.is_on("a") is True  # B keeps its light
    assert camera.whitelamp_calls == [True]

    ctl.find_stop(lit_new, token_new)
    assert ctl.is_on("a") is False


def test_concurrent_searches_first_finisher_leaves_the_lamp_for_the_survivor() -> None:
    # Console and Telegram finders run genuinely concurrently. The one that
    # ends first must not douse the still-running one.
    camera = FakeCamera()
    ctl = control({"a": camera})
    token_a, token_b = object(), object()
    ctl.find_start(["a"], token_a)  # long-running search A
    ctl.find_stop(ctl.find_start(["a"], token_b), token_b)  # B joins and ends fast
    assert ctl.is_on("a") is True  # A still searching, lamp stays

    ctl.find_stop(["a"], token_a)
    assert ctl.is_on("a") is False


def test_ir_hold_placed_before_the_lamp_command_goes_out() -> None:
    # The lamp can light the scene mid-HTTPS-call, so the IR flag must already
    # be frozen when the command is issued — on BOTH lamp-on paths.
    events: list[str] = []

    class OrderedCamera(FakeCamera):
        def reverseWhitelampStatus(self) -> None:
            super().reverseWhitelampStatus()
            events.append(f"lamp:{self.burning}")

    ctl = control({"a": OrderedCamera()}, ir_hold=lambda cam: events.append("hold"))
    ctl.manual_set("a", True)
    assert events[:2] == ["hold", "lamp:True"]

    events.clear()
    ctl2 = control({"b": OrderedCamera()}, ir_hold=lambda cam: events.append("hold"))
    ctl2.find_start(["b"], object())
    assert events[:2] == ["hold", "lamp:True"]


def test_failed_lamp_on_rolls_the_early_hold_back() -> None:
    # A /flash on (or find) against a lamp-less camera must not leave the IR
    # flag frozen forever — the early hold is undone via the delayed release.
    import time

    held: list[str] = []
    released: list[str] = []
    dead = FakeCamera(whitelamp_ok=False, force_ok=False, floodlight_ok=False)
    ctl = control({"a": dead}, ir_hold=held.append, ir_release=released.append)

    assert ctl.manual_set("a", True) is False
    assert held == ["camera-a"]  # the hold went in first...
    deadline = time.monotonic() + 2.0
    while len(released) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert released == ["camera-a"]  # ...and was rolled back

    held.clear()
    released.clear()
    assert ctl.find_start(["a"], object()) == []
    deadline = time.monotonic() + 2.0
    while len(released) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert held == ["camera-a"] and released == ["camera-a"]


def test_reboot_resets_lamp_bookkeeping_and_releases_the_hold() -> None:
    # Power-cycling a flashed camera boots it with the lamp un-forced; the
    # tracked state and the frozen IR flag must not outlive the old lamp,
    # or the sleep tracker/auto-find would see a stuck-IR camera forever.
    released: list[str] = []
    camera = FakeCamera()
    ctl = control({"a": camera}, ir_release=released.append)
    ctl.manual_set("a", True)
    assert ctl.reboot("a") is True
    assert ctl.is_on("a") is False

    import time

    deadline = time.monotonic() + 2.0
    while not released and time.monotonic() < deadline:
        time.sleep(0.01)
    assert released == ["camera-a"]
    # The manual claim died with the reboot too: a later night /find must be
    # able to light this camera again, not skip it as user-owned forever.
    assert ctl.find_start(["a"], object()) == ["a"]


# -- IR hold/release --------------------------------------------------------------


def test_flash_holds_ir_and_releases_after_off() -> None:
    held: list[str] = []
    released: list[str] = []
    camera = FakeCamera()
    ctl = control({"h": camera}, ir_hold=held.append, ir_release=released.append)

    ctl.manual_set("h", True)
    assert held == ["camera-h"]
    assert released == []

    ctl.manual_set("h", False)
    # release_delay is 0 in tests; the timer fires (effectively) immediately.
    import time

    deadline = time.monotonic() + 2.0
    while not released and time.monotonic() < deadline:
        time.sleep(0.01)
    assert released == ["camera-h"]


def test_release_skipped_when_relit_before_timer() -> None:
    held: list[str] = []
    released: list[str] = []
    camera = FakeCamera()
    ctl = TapoControl(
        "pw",
        connect=lambda h, p: camera,
        ir_hold=held.append,
        ir_release=released.append,
        release_delay=0.05,
    )
    ctl.manual_set("h", False)  # schedules a release
    ctl.manual_set("h", True)  # re-lit before the timer fires
    import time

    time.sleep(0.15)
    assert released == []  # the standing hold was preserved


# -- reboot cooldown ----------------------------------------------------------------


def test_reboot_rate_limited_per_host() -> None:
    camera = FakeCamera()
    now = [0.0]
    ctl = control({"h": camera}, clock=lambda: now[0])
    assert ctl.reboot("h") is True
    assert ctl.reboot("h") is False  # inside the cooldown
    now[0] = REBOOT_COOLDOWN_SECONDS + 1.0
    assert ctl.reboot("h") is True
    assert camera.reboots == 2


def test_failed_reboot_still_counts_toward_cooldown() -> None:
    camera = FakeCamera(reboot_ok=False)
    ctl = control({"h": camera})
    assert ctl.reboot("h") is False
    camera.reboot_ok = True
    # Still inside the cooldown: a broken camera is never hammered.
    assert ctl.reboot("h") is False


# -- RebootGuard ----------------------------------------------------------------------


def guard(rebooted: list[str], now: list[float], **kwargs) -> RebootGuard:
    return RebootGuard(
        lambda host: rebooted.append(host) or True,
        wedge_seconds=180.0,
        clock=lambda: now[0],
        spawn=lambda fn, name: fn(),  # run inline so tests are deterministic
        **kwargs,
    )


def test_guard_fires_only_after_sustained_wedge() -> None:
    rebooted: list[str] = []
    now = [0.0]
    g = guard(rebooted, now)
    g.unhealthy("h")  # opens the window
    now[0] = 100.0
    g.unhealthy("h")  # not wedged yet
    assert rebooted == []
    now[0] = 181.0
    g.unhealthy("h")
    assert rebooted == ["h"]


def test_guard_healthy_resets_the_window() -> None:
    rebooted: list[str] = []
    now = [0.0]
    g = guard(rebooted, now)
    g.unhealthy("h")
    now[0] = 179.0
    g.healthy("h")  # a frame arrived just in time
    now[0] = 200.0
    g.unhealthy("h")  # a fresh window opens here
    assert rebooted == []


def test_guard_restarts_window_after_an_attempt() -> None:
    rebooted: list[str] = []
    now = [0.0]
    g = guard(rebooted, now)
    g.unhealthy("h")
    now[0] = 181.0
    g.unhealthy("h")
    assert rebooted == ["h"]
    now[0] = 200.0
    g.unhealthy("h")  # camera still down while it boots — no double fire
    assert rebooted == ["h"]
    now[0] = 181.0 + 181.0
    g.unhealthy("h")
    assert rebooted == ["h", "h"]


def test_guard_notifies_with_downtime() -> None:
    notified: list[tuple[str, float]] = []
    now = [0.0]
    g = guard([], now, notify=lambda host, downtime: notified.append((host, downtime)))
    g.unhealthy("h")
    now[0] = 240.0
    g.unhealthy("h")
    assert notified == [("h", 240.0)]


def test_guard_forget_clears_a_stale_window() -> None:
    # A camera retired for hours (watchlist/DHCP) must not be instantly
    # rebooted with a bogus downtime the moment it is re-added.
    rebooted: list[str] = []
    now = [0.0]
    g = guard(rebooted, now)
    g.unhealthy("h")
    g.forget("h")  # the supervisor retired the camera
    now[0] = 7200.0
    g.unhealthy("h")  # re-added much later: a fresh window, no reboot
    assert rebooted == []


def test_guard_forget_survives_an_in_flight_reboot() -> None:
    # The camera is retired WHILE a reboot attempt is in flight. The attempt's
    # window-restart must not resurrect the forgotten window, or the host
    # would be instantly re-rebooted when re-added hours later.
    rebooted: list[str] = []
    now = [0.0]
    pending: list = []
    g = RebootGuard(
        lambda host: rebooted.append(host) or True,
        wedge_seconds=180.0,
        clock=lambda: now[0],
        spawn=lambda fn, name: pending.append(fn),  # defer: reboot "in flight"
    )
    g.unhealthy("h")
    now[0] = 181.0
    g.unhealthy("h")  # spawns the deferred reboot
    g.forget("h")  # the supervisor retires the camera mid-flight
    pending.pop()()  # the reboot attempt completes now
    assert rebooted == ["h"]

    now[0] = 7200.0
    g.unhealthy("h")  # re-added hours later: fresh window, no instant reboot
    assert rebooted == ["h"]
    notified: list[tuple[str, float]] = []
    now = [0.0]
    g = RebootGuard(
        lambda host: False,
        wedge_seconds=180.0,
        clock=lambda: now[0],
        spawn=lambda fn, name: fn(),
        notify=lambda host, downtime: notified.append((host, downtime)),
    )
    g.unhealthy("h")
    now[0] = 181.0
    g.unhealthy("h")
    assert notified == []


# -- FindFlash --------------------------------------------------------------------------


def test_find_flash_lights_only_ir_cameras_and_reports() -> None:
    a, b = FakeCamera(), FakeCamera()
    ctl = control({"a": a, "b": b})
    flash = FindFlash(
        ctl,
        ir_cameras=lambda: {"camera-a"},  # only a is dark
        hosts=lambda: {"a", "b"},
        display=lambda name: name.removeprefix("camera-").upper(),
    )
    note = flash.start()
    assert note is not None and "A" in note and "spotlight" in note.lower()
    assert a.whitelamp_calls == [True]
    assert b.whitelamp_calls == []

    flash.stop()
    assert a.whitelamp_calls == [True, False]


def test_find_flash_quiet_when_nothing_is_dark() -> None:
    ctl = control({"a": FakeCamera()})
    flash = FindFlash(ctl, ir_cameras=lambda: set(), hosts=lambda: {"a"})
    assert flash.start() is None
    flash.stop()  # nothing lit; a no-op


def test_find_flash_stop_is_idempotent() -> None:
    camera = FakeCamera()
    ctl = control({"a": camera})
    flash = FindFlash(ctl, ir_cameras=lambda: {"camera-a"}, hosts=lambda: {"a"})
    flash.start()
    flash.stop()
    flash.stop()
    assert camera.whitelamp_calls == [True, False]  # off exactly once


# -- camera matching ---------------------------------------------------------------------


def test_match_camera_hosts_by_ip_octet_and_name() -> None:
    hosts = ["192.168.1.19", "192.168.1.190", "192.168.1.22"]
    display = {
        "camera-192.168.1.19": "Cockatiel Tower",
        "camera-192.168.1.190": "Back Wall",
        "camera-192.168.1.22": "Sunset Tower",
    }
    lookup = lambda name: display[name]
    assert match_camera_hosts("192.168.1.19", hosts, lookup) == ["192.168.1.19"]
    # A last-octet shorthand must not also catch .190.
    assert match_camera_hosts(".19", hosts, lookup) == ["192.168.1.19"]
    assert match_camera_hosts("19", hosts, lookup) == ["192.168.1.19"]
    assert match_camera_hosts("cockatiel", hosts, lookup) == ["192.168.1.19"]
    # A fragment can address several cameras at once.
    assert match_camera_hosts("tower", hosts, lookup) == ["192.168.1.19", "192.168.1.22"]
    assert match_camera_hosts("nope", hosts, lookup) == []
    assert match_camera_hosts("", hosts, lookup) == []


def test_match_camera_hosts_numeric_never_falls_through_to_names() -> None:
    # An un-named camera's display falls back to something IP-derived; a bare
    # "19" must still address only the .19 camera, not every name with a 19.
    hosts = ["192.168.1.19", "192.168.1.190"]
    lookup = lambda name: name  # display fallback: "camera-192.168.1.190"
    assert match_camera_hosts("19", hosts, lookup) == ["192.168.1.19"]
    assert match_camera_hosts("190", hosts, lookup) == ["192.168.1.190"]


def test_lamp_state_prefers_device_truth_and_falls_back_to_tracked() -> None:
    # After a server restart the tracked state starts at off while the lamp
    # may physically still burn — the device read must win.
    camera = FakeCamera()
    camera.burning = True  # the lamp was lit pre-restart
    ctl = control({"a": camera})
    assert ctl.is_on("a") is False  # tracked state knows nothing
    assert ctl.lamp_state("a") is True  # the device tells the truth

    # An unreachable camera degrades to the tracked guess.
    gone = control({})
    assert gone.lamp_state("gone") is False
