"""
Abstract base interface for wake word detection.

All wake word detectors implement ``BaseWakeWordDetector`` so they can
be swapped transparently between OpenWakeWord (production) and
text-matching (fallback/legacy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from speech.models.audio_frame import AudioFrame


class WakeWordResult:
    """Result of processing a frame through a wake word detector.

    Attributes:
        detected:   True if the wake word was detected in this frame.
        confidence: Detection confidence [0.0, 1.0].
        keyword:    The detected wake word string (e.g. "simon").
    """

    __slots__ = ("detected", "confidence", "keyword")

    def __init__(
        self,
        detected: bool = False,
        confidence: float = 0.0,
        keyword: str = "",
    ):
        self.detected = detected
        self.confidence = confidence
        self.keyword = keyword

    def __repr__(self) -> str:
        return f"WakeWordResult(detected={self.detected}, conf={self.confidence:.2f}, kw={self.keyword!r})"


class BaseWakeWordDetector(ABC):
    """Abstract base class for wake word detectors."""

    @abstractmethod
    def process_frame(self, frame: AudioFrame) -> WakeWordResult:
        """Process an audio frame and check for wake word.

        Args:
            frame: Audio frame to analyse.

        Returns:
            ``WakeWordResult`` with detection status.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this wake word detector implementation."""
        ...

    @property
    @abstractmethod
    def wake_word(self) -> str:
        """The configured wake word."""
        ...

    def set_threshold(self, threshold: float) -> None:
        """Update the detection threshold (for adaptive control).

        Default no-op; subclasses override.
        """
        pass
