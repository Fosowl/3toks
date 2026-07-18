"""Offline camera tests: never open real hardware, never import cv2.

camera_available() must answer without importing cv2, and the cv2-missing
path of capture_jpeg_b64 is asserted ONLY when find_spec confirms cv2 is
absent (otherwise skipped, so no real device is ever opened).
"""
import importlib.util
import unittest

from threetoks import camera

_CV2_PRESENT = importlib.util.find_spec("cv2") is not None


class CameraAvailableTest(unittest.TestCase):
    def test_available_returns_bool_without_raising(self):
        self.assertIsInstance(camera.camera_available(), bool)

    def test_available_agrees_with_find_spec(self):
        self.assertEqual(camera.camera_available(), _CV2_PRESENT)


class CaptureImportGuardTest(unittest.TestCase):
    @unittest.skipIf(_CV2_PRESENT, "cv2 present: would open real hardware")
    def test_capture_returns_none_when_cv2_missing(self):
        self.assertIsNone(camera.capture_jpeg_b64())


if __name__ == "__main__":
    unittest.main()
