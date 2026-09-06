"""Abstract base class for OCR engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OCRResult:
    """A single OCR text detection result.

    Attributes
    ----------
    text : str
        The recognized text content.
    confidence : float
        Recognition confidence [0.0, 1.0].
    bbox : tuple[int, int, int, int] | None
        Bounding box ``(x1, y1, x2, y2)`` of the text region.
    """

    text: str
    confidence: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None


class BaseOCREngine(abc.ABC):
    """Abstract OCR engine.

    Subclasses must implement ``read_text()`` and ``is_ready()``.
    """

    @abc.abstractmethod
    def read_text(self, frame: Any) -> list[OCRResult]:
        """Extract text from an image frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image (OpenCV convention).

        Returns
        -------
        list[OCRResult]
            Detected text regions with confidence.
        """
        ...

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return True if the OCR engine is loaded and ready."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
