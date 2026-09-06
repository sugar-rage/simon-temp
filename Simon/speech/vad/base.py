"""
Abstract base interface for Voice Activity Detection (VAD).

All VAD implementations must inherit from ``BaseVAD`` and implement
``process_frame()``.  This allows transparent swapping between Silero VAD
(production) and energy-based VAD (fallback) without touching consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from speech.models.audio_frame import AudioFrame


class VADResult:
    """Result of processing a single audio frame through a VAD.

    Attributes:
        is_speech:    True if the frame contains speech.
        confidence:   Speech probability [0.0, 1.0].
        speech_start: True if this frame is the START of a new speech segment.
        speech_end:   True if this frame is the END of a speech segment.
    """

    __slots__ = ("is_speech", "confidence", "speech_start", "speech_end")

    def __init__(
        self,
        is_speech: bool = False,
        confidence: float = 0.0,
        speech_start: bool = False,
        speech_end: bool = False,
    ):
        self.is_speech = is_speech
        self.confidence = confidence
        self.speech_start = speech_start
        self.speech_end = speech_end

    def __repr__(self) -> str:
        return (
            f"VADResult(speech={self.is_speech}, conf={self.confidence:.2f}, "
            f"start={self.speech_start}, end={self.speech_end})"
        )


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection."""

    @abstractmethod
    def process_frame(self, frame: AudioFrame) -> VADResult:
        """Process a single audio frame and return a VAD decision.

        Args:
            frame: Audio frame to analyse.

        Returns:
            A ``VADResult`` indicating whether speech is present.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""
        ...

    @abstractmethod
    def get_speech_segment(self) -> Optional[AudioFrame]:
        """Return the accumulated speech segment if one is complete.

        Returns:
            An ``AudioFrame`` containing the full speech segment,
            or ``None`` if no complete segment is available yet.
        """
        ...

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """True if currently in a speech segment."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this VAD implementation."""
        ...

    def set_threshold(self, threshold: float) -> None:
        """Update the speech detection threshold (for adaptive control).

        Default implementation is a no-op; subclasses override if supported.
        """
        pass
