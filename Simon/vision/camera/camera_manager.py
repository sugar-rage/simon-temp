"""Camera manager with automatic reconnection and frame buffering.

Owns the camera lifecycle and provides a thread-safe ``get_frame()``
method for the vision pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

from core.models.frame import FrameData
from core.config.system_config import CameraConfig
from vision.camera.base import BaseCameraSource

logger = logging.getLogger("simon.vision.camera")


class CameraManager:
    """Manages camera lifecycle, auto-reconnection, and frame buffering.

    Parameters
    ----------
    camera : BaseCameraSource
        The camera source to manage.
    config : CameraConfig
        Camera configuration.
    """

    def __init__(
        self,
        camera: BaseCameraSource,
        config: Optional[CameraConfig] = None,
    ) -> None:
        self._camera = camera
        self._config = config or CameraConfig()
        self._buffer: deque[FrameData] = deque(
            maxlen=self._config.frame_buffer_size
        )
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Open the camera and start the background capture thread."""
        if self._running:
            return True

        if not self._camera.open():
            logger.error("Camera failed to open")
            return False

        self._connected = True
        self._running = True
        self._reconnect_attempts = 0

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("Camera manager started")
        return True

    def stop(self) -> None:
        """Stop capture and release the camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
        self._camera.release()
        self._connected = False
        with self._lock:
            self._buffer.clear()
        logger.info("Camera manager stopped")

    def get_frame(self) -> Optional[FrameData]:
        """Return the most recent frame, or None if buffer is empty."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def get_all_frames(self) -> list[FrameData]:
        """Return all buffered frames (oldest first)."""
        with self._lock:
            return list(self._buffer)

    def _capture_loop(self) -> None:
        """Background frame capture loop with auto-reconnection."""
        while self._running:
            frame = self._camera.read()

            if frame is not None and frame.is_valid:
                with self._lock:
                    self._buffer.append(frame)
                self._reconnect_attempts = 0
                if not self._connected:
                    self._connected = True
                    logger.info("Camera reconnected")
            else:
                self._connected = False
                if self._config.auto_reconnect:
                    self._attempt_reconnect()
                else:
                    logger.warning("Frame capture failed, auto-reconnect disabled")
                    time.sleep(0.1)

            # Maintain target FPS
            time.sleep(1.0 / max(self._config.fps, 1))

    def _attempt_reconnect(self) -> None:
        """Try to reconnect the camera with exponential backoff."""
        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(
                "Max reconnection attempts (%d) exceeded",
                self._max_reconnect_attempts,
            )
            self._running = False
            return

        delay = min(
            self._config.reconnect_delay_s * (2 ** (self._reconnect_attempts - 1)),
            30.0,
        )
        logger.warning(
            "Camera disconnected, reconnect attempt %d/%d in %.1fs",
            self._reconnect_attempts,
            self._max_reconnect_attempts,
            delay,
        )
        time.sleep(delay)

        self._camera.release()
        if self._camera.open():
            self._connected = True
            self._reconnect_attempts = 0
            logger.info("Camera reconnected successfully")
