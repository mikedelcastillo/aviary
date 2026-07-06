"""aviary RGB — turn the PC's motherboard / header / strip LEDs into a live
status display for the aviary (discovery sweeps, bird sightings, IR night mode).

Entry point for the server: :func:`build_controller`, which reads the RGB_* app
config and returns a started-elsewhere :class:`RGBController` (or ``None`` when
disabled). The controller is fed state via ``on_discovery`` / ``on_detection`` /
``on_ir`` and exposes ``command`` for the ``/rgb`` Telegram + console command.
"""
from __future__ import annotations

import logging

from .engine import RGBController
from .surface import SurfaceConfig

log = logging.getLogger("aviary.rgb")

__all__ = ["RGBController", "build_controller"]


def build_controller(stop_event, app_config, ir_state=None) -> RGBController | None:
    """Construct an RGBController from AppConfig, or None if RGB is disabled.

    The caller starts it (``.start()``), registers ``ir_state.add_listener(
    controller.on_ir)``, wires ``on_discovery``/``on_detection``, and stops it in
    the shutdown path. Never raises: any failure logs and returns None so the
    server runs fine without lights.
    """
    if not getattr(app_config, "rgb_enabled", False):
        return None
    try:
        cfg = SurfaceConfig(
            host=getattr(app_config, "rgb_host", "127.0.0.1") or "127.0.0.1",
            port=int(getattr(app_config, "rgb_port", 6742) or 6742),
            addressable_len=int(getattr(app_config, "rgb_strip_len", -1)),
        )
        controller = RGBController(
            stop_event,
            brightness=float(getattr(app_config, "rgb_brightness", 1.0) or 1.0),
            ir_state=ir_state,
            surface_config=cfg,
            night_led=int(getattr(app_config, "rgb_night_led", -1)),
        )
        _log_roster_warnings()
        return controller
    except Exception:  # noqa: BLE001
        log.exception("Failed to build RGB controller; lights disabled.")
        return None


def _log_roster_warnings() -> None:
    try:
        from lib.roster import load_species_members

        from . import birds

        for w in birds.validate_against(load_species_members()):
            log.warning("RGB roster: %s", w)
    except Exception:  # noqa: BLE001
        pass
