"""
Abstract base interface for Speech-to-Text engines.

All STT engines (faster-whisper, SpeechBrain, Vosk, etc.) must implement
``BaseSTTEngine`` so that the engine orchestrator can manage them
interchangeably.

Design decision: The interface uses both batch ``transcribe()`` and
streaming ``transcribe_streaming()`` methods.  Engines that don't support
streaming can raise ``NotImplementedError`` — the orchestrator will
fall back to batch mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.stt.transcript import Transcript


class BaseSTTEngine(ABC):
    """Abstract base class for STT engines.

    Provides a uniform interface for batch and streaming transcription,
    model lifecycle management, and engine metadata.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: AudioFrame,
        params: Optional[DecodingParams] = None,
    ) -> Transcript:
        """Transcribe a complete audio segment.

        Args:
            audio:  AudioFrame containing the speech to transcribe.
            params: Decoding parameters (beam size, temperature, etc.).
                    If None, engine-specific defaults are used.

        Returns:
            A ``Transcript`` with the recognised text and metadata.
        """
        ...

    def transcribe_streaming(
        self,
        audio_iter: Iterator[AudioFrame],
        params: Optional[DecodingParams] = None,
        callback: Optional[Callable[[Transcript], None]] = None,
    ) -> Transcript:
        """Transcribe a stream of audio frames with partial results.

        Default implementation collects all frames and calls ``transcribe()``.
        Engines with native streaming support should override this.

        Args:
            audio_iter: Iterator yielding AudioFrame chunks.
            params:     Decoding parameters.
            callback:   Called with partial Transcript results as they arrive.

        Returns:
            The final Transcript.
        """
        frames = list(audio_iter)
        if not frames:
            return Transcript()
        combined = AudioFrame.concatenate(frames)
        return self.transcribe(combined, params)

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (may download on first call)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (free GPU memory, etc.)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this engine (e.g. 'faster-whisper')."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True if the model is loaded and ready for transcription."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """True if this engine supports native streaming transcription."""
        return False

    @property
    def supports_word_timestamps(self) -> bool:
        """True if this engine can provide per-word timestamps."""
        return False
