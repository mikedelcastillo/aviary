from __future__ import annotations

from lib.ir import IRState


def test_ir_cameras_and_all_ir() -> None:
    ir = IRState()
    assert ir.all_ir() is False  # nothing reported yet
    ir.update("a", True)
    ir.update("b", False)
    assert ir.ir_cameras() == {"a"}
    assert ir.is_ir("a") and not ir.is_ir("b")
    assert ir.all_ir() is False
    ir.update("b", True)
    assert ir.all_ir() is True
    assert ir.known() == {"a", "b"}


def test_listeners_fire_only_on_transition() -> None:
    ir = IRState()
    events: list[tuple[str, bool]] = []
    ir.add_listener(lambda cam, on: events.append((cam, on)))
    ir.update("a", True)   # transition None->True
    ir.update("a", True)   # no change -> no event
    ir.update("a", False)  # transition -> event
    assert events == [("a", True), ("a", False)]


def test_forget_drops_camera() -> None:
    ir = IRState()
    ir.update("a", True)
    ir.update("b", True)
    assert ir.all_ir() is True
    ir.forget("b")
    assert ir.known() == {"a"} and ir.all_ir() is True
