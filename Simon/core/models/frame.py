"""FrameData — camera frame container with metadata.

Wraps a numpy array with capture metadata (timestamp, resolution,
source ID, frame counter) for consistent frame handling across the
vision pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FrameData:
    """A single camera frame with capture metadata.

    Parameters
    ----------
    data : numpy.ndarray
        BGR image data (OpenCV convention).
    timestamp : float
        Capture time (``time.time()``).  Defaults to now.
    frame_id : int
        Monotonically increasing frame counter.
    resolution : tuple[int, int]
        ``(width, height)`` in pixels.
    source_id : str
        Camera device identifier.
    """

    data: Any  # np.ndarray — typed as Any to avoid hard numpy import
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    resolution: tuple[int, int] = (640, 480)
    source_id: str = "default"

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def is_valid(self) -> bool:
        """Return True if this frame contains usable image data."""
        return self.data is not None and hasattr(self.data, "shape")
