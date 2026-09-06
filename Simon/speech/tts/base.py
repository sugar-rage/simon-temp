"""
Abstract interfaces for the Text-to-Speech (TTS) subsystem.
"""

from __future__ import annotations

import abc
from typing import Iterator

from speech.models.audio_frame import AudioFrame
from speech.tts.emotional_profile import EmotionalStyle


class BaseTTSEngine(abc.ABC):
    """Abstract base class for all TTS engines (Neural, Pyttsx3, etc.)."""

    @abc.abstractmethod
    def synthesize(self, text: str, style: EmotionalStyle) -> Iterator[AudioFrame]:
        """Synthesize text into audio frames.
        
        This method must be a generator that yields `AudioFrame` instances.
        For streaming engines, it yields chunks as soon as they are ready.
        For block-based engines, it may yield a single large frame.
        
        Args:
            text: The text to synthesize.
            style: The emotional style (speed, pitch, volume) to apply.
            
        Yields:
            AudioFrame: Chunks of synthesized audio.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the TTS engine."""
        pass

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Check if the engine models are loaded into memory."""
        pass

    @abc.abstractmethod
    def load(self) -> None:
        """Load models into memory. Should be idempotent."""
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload models from memory to free resources."""
        pass
