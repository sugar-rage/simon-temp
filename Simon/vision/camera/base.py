"""Abstract base class for camera sources.

All camera implementations (OpenCV, PiCamera, simulated) extend this ABC.
"""

from __future__ import annotations

import abc
from typing import Optional

from core.models.frame import FrameData


class BaseCameraSource(abc.ABC):
    """Abstract camera source.

    Subclasses must implement ``open()``, ``read()``, and ``release()``.
    """

    @abc.abstractmethod
    def open(self) -> bool:
        """Open the camera device.  Return True on success."""
        ...

    @abc.abstractmethod
    def read(self) -> Optional[FrameData]:
        """Capture a single frame.  Return None on failure."""
        ...

    @abc.abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        ...

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Return True if the camera is currently open."""
        ...

    @property
    def name(self) -> str:
        """Human-readable camera source name."""
        return self.__class__.__name__
