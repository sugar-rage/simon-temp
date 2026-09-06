"""OpenCV-based camera source.

Wraps ``cv2.VideoCapture`` with automatic reconnection, frame ID tracking,
and resolution configuration.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Optional

from core.models.frame import FrameData
from core.config.system_config import CameraConfig
from vision.camera.base import BaseCameraSource

logger = logging.getLogger("simon.vision.camera")


class CV2Camera(BaseCameraSource):
    """Camera source using OpenCV's VideoCapture.

    Parameters
    ----------
    config : CameraConfig
        Camera configuration (device index, resolution, FPS).
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._config = config or CameraConfig()
        self._cap: Any = None  # cv2.VideoCapture — typed as Any
        self._frame_counter = 0
        self._opened = False

    def open(self) -> bool:
        """Open the camera device."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is not installed")
            return False

        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self._config.device_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._config.device_index)
        if not self._cap.isOpened():
            logger.error(
                "Failed to open camera device %d", self._config.device_index
            )
            self._opened = False
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        self._opened = True
        self._frame_counter = 0
        logger.info(
            "Camera opened: device=%d, resolution=%dx%d, fps=%d",
            self._config.device_index,
            self._config.width,
            self._config.height,
            self._config.fps,
        )
        return True

    def read(self) -> Optional[FrameData]:
        """Capture a single frame."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Frame capture failed")
            return None

        self._frame_counter += 1
        h, w = frame.shape[:2]

        return FrameData(
            data=frame,
            timestamp=time.time(),
            frame_id=self._frame_counter,
            resolution=(w, h),
            source_id=f"cv2:{self._config.device_index}",
        )

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False
        logger.info("Camera released")

    def is_opened(self) -> bool:
        """Return True if the camera is currently open."""
        if self._cap is None:
            return False
        return self._opened and self._cap.isOpened()
