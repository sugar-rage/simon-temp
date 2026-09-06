"""Abstract base class for object detectors."""

from __future__ import annotations

import abc
from typing import Any

from core.models.detection import Detection


class BaseDetector(abc.ABC):
    """Abstract object detector.

    Subclasses must implement ``detect()`` and ``is_ready()``.
    """

    @abc.abstractmethod
    def detect(self, frame: Any) -> list[Detection]:
        """Run detection on a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image (OpenCV convention).

        Returns
        -------
        list[Detection]
            Detected objects with bounding boxes and confidence scores.
        """
        ...

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return True if the detector model is loaded and ready."""
        ...

    @property
    def name(self) -> str:
        """Human-readable detector name."""
        return self.__class__.__name__
