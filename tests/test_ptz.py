from __future__ import annotations

import base64
import hashlib

from lib.ptz import (
    OnvifPatrol,
    PATROL_LEG_STEPS,
    absolute_move_body,
    build_envelope,
    build_wss_header,
    capabilities_have_ptz,
    continuous_move_body,
    parse_pantilt_position,
    parse_profile_token,
)


# -- SOAP / WS-Security construction ---------------------------------------


def test_wss_header_uses_password_digest_formula() -> None:
    nonce = b"0123456789abcdef"
    created = "2020-01-01T00:00:00Z"
    header = build_wss_header("alice", "s3cret", nonce=nonce, created=created)

    expected_digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + b"s3cret").digest()
    ).decode()
    assert expected_digest in header
    assert "<Username>alice</Username>" in header
    assert base64.b64encode(nonce).decode() in header
    assert created in header


def test_envelope_wraps_header_and_body() -> None:
    env = build_envelope("<H/>", "<B/>")
    assert env.startswith("<?xml")
    assert "<s:Header><H/></s:Header>" in env
    assert "<s:Body><B/></s:Body>" in env


def test_continuous_and_absolute_move_bodies_carry_values() -> None:
    move = continuous_move_body("profile_1", 0.4, -0.1)
    assert "profile_1" in move
    assert 'x="0.4"' in move and 'y="-0.1"' in move
    absolute = absolute_move_body("profile_1", 0.2, -0.7)
    assert "AbsoluteMove" in absolute
    assert 'x="0.2"' in absolute and 'y="-0.7"' in absolute


# -- response parsing -------------------------------------------------------


def test_capabilities_have_ptz_detects_ptz_node() -> None:
    assert capabilities_have_ptz("<tt:PTZ><tt:XAddr>http://x/ptz</tt:XAddr></tt:PTZ>")
    assert capabilities_have_ptz("<PTZ >...")
    assert not capabilities_have_ptz("<tt:Media><tt:XAddr>http://x</tt:XAddr></tt:Media>")


def test_parse_profile_token() -> None:
    assert parse_profile_token('<Profiles token="profile_1" fixed="true">') == "profile_1"
    assert parse_profile_token("<Profiles>no token</Profiles>") is None


def test_parse_pantilt_position() -> None:
    text = '<tt:Position><tt:PanTilt x="0.207243" y="-0.705128"></tt:PanTilt></tt:Position>'
    assert parse_pantilt_position(text) == (0.207243, -0.705128)
    assert parse_pantilt_position("<tt:Position/>") is None


# -- patrol behaviour -------------------------------------------------------


class FakeCamera:
    def __init__(self, host: str, position=(0.1, 0.2)) -> None:
        self.host = host
        self._position = position
        self.moves: list[float] = []
        self.stopped = 0
        self.restored: list[tuple[float, float]] = []
        self.homed = 0

    def get_position(self):
        return self._position

    def move(self, pan: float, tilt: float = 0.0) -> bool:
        self.moves.append(pan)
        return True

    def stop(self) -> bool:
        self.stopped += 1
        return True

    def absolute_move(self, pan: float, tilt: float) -> bool:
        self.restored.append((pan, tilt))
        return True

    def goto_home(self) -> bool:
        self.homed += 1
        return True


def test_patrol_sweeps_then_reverses() -> None:
    camera = FakeCamera("192.168.1.8")
    patrol = OnvifPatrol([camera], pan_speed=0.4)
    patrol.start()
    for _ in range(PATROL_LEG_STEPS * 2):
        patrol.step()
    # First leg pans one way, second leg the other.
    assert camera.moves[:PATROL_LEG_STEPS] == [0.4] * PATROL_LEG_STEPS
    assert camera.moves[PATROL_LEG_STEPS:] == [-0.4] * PATROL_LEG_STEPS


def test_patrol_restores_exact_position_on_stop() -> None:
    camera = FakeCamera("192.168.1.8", position=(0.207, -0.705))
    patrol = OnvifPatrol([camera])
    patrol.start()
    patrol.step()
    patrol.stop()
    assert camera.stopped == 1
    # Returns to the EXACT pre-patrol facing, not a generic home.
    assert camera.restored == [(0.207, -0.705)]
    assert camera.homed == 0


def test_patrol_falls_back_to_home_when_position_unknown() -> None:
    class NoPositionCamera(FakeCamera):
        def get_position(self):
            return None

    camera = NoPositionCamera("192.168.1.30")
    patrol = OnvifPatrol([camera])
    patrol.start()
    patrol.stop()
    # No captured position -> fall back to the camera's home preset.
    assert camera.homed == 1
    assert camera.restored == []
