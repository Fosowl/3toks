"""Webcam capture: one JPEG frame as raw base64 for vision decisions.

``cv2`` (opencv-python) is an optional extra. This module stays
stdlib-only at import time and imports cv2 lazily inside the one
function that needs it, so core imports never require it (the
lazy-extras rule, enforced by tests/test_lazy_extras.py).
"""
import base64
import importlib.util

WARMUP_FRAMES = 5           # discarded so auto-exposure can settle
JPEG_EXT = ".jpg"


def camera_available() -> bool:
    """Whether opencv-python (cv2) is importable, without importing it.

    Returns True if the ``cv2`` module can be found on the import path,
    False otherwise. Has no cv2 import side effects.
    """
    try:
        return importlib.util.find_spec("cv2") is not None
    except ModuleNotFoundError:
        return False


def capture_jpeg_b64(index: int = 0) -> str | None:
    """Grab one webcam frame as a raw base64 JPEG string.

    ``index`` selects the capture device. Returns the base64 text with
    NO data-URI prefix (backends add their own framing), or None when
    cv2 is missing, the device will not open, or the read/encode fails.
    """
    try:
        import cv2  # optional extra, imported lazily
    except ImportError:
        return None
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    try:
        return _grab_frame(cap, cv2)
    finally:
        cap.release()          # always free the device


def _grab_frame(cap, cv2) -> str | None:
    """Warm up, read one frame, return it base64-encoded or None."""
    for _ in range(WARMUP_FRAMES):
        cap.read()             # discard while exposure settles
    read_ok, frame = cap.read()
    if not read_ok or frame is None:
        return None
    enc_ok, buffer = cv2.imencode(JPEG_EXT, frame)
    if not enc_ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


if __name__ == "__main__":
    assert isinstance(camera_available(), bool)
    if not camera_available():          # no cv2 -> capture must be None
        assert capture_jpeg_b64() is None
    print("smoke OK")
