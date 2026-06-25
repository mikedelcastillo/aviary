from __future__ import annotations

from lib.ai.vlm import clean_camera_name, describe_scene, encode_image, name_camera_view


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


def test_describe_scene_sends_image_to_model() -> None:
    client = FakeClient("Percy is on the perch with Matcha.")
    out = describe_scene(client, "qwen2.5vl:7b", b"jpegbytes")
    assert out == "Percy is on the perch with Matcha."
    assert client.calls[0]["images"] is not None
    assert client.calls[0]["model"] == "qwen2.5vl:7b"


def test_name_camera_view_cleans_model_output() -> None:
    client = FakeClient("Sure! The label is: Window Perch.")
    assert name_camera_view(client, "qwen2.5vl:7b", b"jpeg") == "Sure The"
