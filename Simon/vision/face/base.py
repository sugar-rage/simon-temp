"""Abstract base class for face recognizers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FaceResult:
    """A single face recognition result.

    Attributes
    ----------
    name : str
        Recognized identity or ``"Unknown"``.
    confidence : float
        Recognition confidence [0.0, 1.0].
    bbox : tuple[int, int, int, int]
        Face bounding box ``(x1, y1, x2, y2)``.
    embedding : Any
        Face embedding vector (for database operations).
    """

    name: str = "Unknown"
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    embedding: Any = None


class BaseFaceRecognizer(abc.ABC):
    """Abstract face recognizer.

    Subclasses must implement ``detect_faces()`` and ``is_ready()``.
    """

    @abc.abstractmethod
    def detect_faces(self, frame: Any) -> list[FaceResult]:
        """Detect and recognize faces in a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image.

        Returns
        -------
        list[FaceResult]
            Detected faces with identity and confidence.
        """
        ...

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """Return True if the face recognition model is loaded."""
        ...

    @abc.abstractmethod
    def save_face(self, frame: Any, face: FaceResult, name: str) -> bool:
        """Save a face to the identity database.

        Returns True on success.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
