from __future__ import annotations

import json
from datetime import datetime, timezone

from lib.activity_qa import ActivityResponder

KNOWN = ["bambi", "draft", "matcha", "percy", "pizza"]


def _write_sighting(collect_dir, label, conf, when_epoch) -> None:
    folder = collect_dir / label
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{int(when_epoch)}"
    (folder / f"{stem}.jpg").write_bytes(b"\xff\xd8jpeg")
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {
                "object": label,
                "camera": {"name": "camera-192.168.1.8"},
                "collected_at": datetime.fromtimestamp(when_epoch, timezone.utc).isoformat(),
                "frame": {"width": 100, "height": 100},
                "detection": {"confidence": conf, "bbox_xyxy": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}},
            }
        )
    )


class FakeClient:
    def generate(self, model, prompt, *, images=None, timeout_seconds=None, **kwargs):
        return "perched and preening"

    def chat(self, model, messages, **kwargs):
        return "Percy spent the afternoon preening on the perch."


def _responder(collect_dir, notify, find=None, send_photo=None, now=2000.0):
    return ActivityResponder(
        collect_dir,
        FakeClient(),
        "qwen3:4b",
        "qwen2.5vl:7b",
        lambda: KNOWN,
        notify=notify,
        send_photo=send_photo,
        find=find,
        camera_display=lambda name: "Big Cage",
        clock=lambda: now,
    )


def test_respond_summarises_a_birds_day(tmp_path) -> None:
    _write_sighting(tmp_path, "percy", 0.9, 1900)
    sent: list = []
    photos: list = []
    _responder(
        tmp_path,
        lambda c, t: sent.append(t),
        send_photo=lambda c, img, cap: photos.append((img, cap)),
    ).respond(7, "what did percy do today?", "percy")
    assert any("preening" in t for t in sent)
    assert len(photos) >= 1


def test_respond_triggers_find_when_unseen_and_live(tmp_path) -> None:
    found: list = []
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: found.append((cid, arg))).respond(
        7, "what is draft doing right now?", "draft"
    )
    # No recent draft sighting + a "now" question -> go look live.
    assert found == [(7, "draft")]
    assert any("check the cameras" in t for t in sent)


def test_respond_says_unseen_when_not_live(tmp_path) -> None:
    sent: list = []
    _responder(tmp_path, lambda c, t: sent.append(t), find=lambda cid, arg: None).respond(
        7, "did matcha appear today?", "matcha"
    )
    assert sent and "haven't seen" in sent[0].lower()
