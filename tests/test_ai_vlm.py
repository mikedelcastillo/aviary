from __future__ import annotations

from dataclasses import dataclass

from lib.ai.vlm import (
    build_detection_context,
    clean_camera_name,
    describe_scene,
    encode_image,
    name_camera_view,
    position_phrase,
)


@dataclass
class FakeDetection:
    label: str
    bbox_xyxy: tuple[int, int, int, int]


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, model, prompt, *, images=None, timeout_seconds=None, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, "images": images})
        return self.response


def test_encode_image_is_base64() -> None:
    import base64

    assert base64.b64decode(encode_image(b"hello")) == b"hello"


def test_clean_camera_name_normalises() -> None:
    assert clean_camera_name('"Window Perch"') == "Window Perch"
    assert clean_camera_name("food bowl, top view") == "Food Bowl"
    assert clean_camera_name("The Big Cage\nextra line") == "The Big"
    assert clean_camera_name("") == ""


def test_position_phrase_thirds() -> None:
    assert position_phrase(10, 10, 100, 100) == "top-left"
    assert position_phrase(50, 50, 100, 100) == "centre"
    assert position_phrase(90, 90, 100, 100) == "bottom-right"
    assert position_phrase(50, 10, 100, 100) == "top-centre"


def test_build_detection_context_grounds_the_vlm() -> None:
    detections = [
        FakeDetection("percy", (0, 0, 20, 20)),       # top-left
        FakeDetection("pizza", (180, 180, 200, 200)),  # bottom-right
    ]
    context = build_detection_context(detections, 200, 200, {"pizza": "cockatiel"})
    assert "Percy in the top-left" in context
    assert "Pizza (a cockatiel) in the bottom-right" in context


def test_build_detection_context_when_empty() -> None:
    context = build_detection_context([], 200, 200)
    assert "no birds" in context.lower()


def test_describe_scene_prepends_context() -> None:
    client = FakeClient("Percy is on the perch with Matcha.")
    out = describe_scene(client, "qwen2.5vl:7b", b"jpegbytes", context="Detected: Percy top-left.")
    assert out == "Percy is on the perch with Matcha."
    assert "Detected: Percy top-left." in client.calls[0]["prompt"]
    assert client.calls[0]["model"] == "qwen2.5vl:7b"


def test_name_camera_view_cleans_model_output() -> None:
    client = FakeClient("Sure! The label is: Window Perch.")
    assert name_camera_view(client, "qwen2.5vl:7b", b"jpeg") == "Sure The"
