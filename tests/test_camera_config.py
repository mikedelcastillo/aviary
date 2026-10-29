from lib.camera import configure_ffmpeg_capture


def test_decoder_thread_default_and_operator_override(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_THREADS", raising=False)
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    configure_ffmpeg_capture()
    import os
    assert os.environ["OPENCV_FFMPEG_THREADS"] == "2"
    assert "rtsp_transport;tcp" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    monkeypatch.setenv("OPENCV_FFMPEG_THREADS", "4")
    configure_ffmpeg_capture()
    assert os.environ["OPENCV_FFMPEG_THREADS"] == "4"
