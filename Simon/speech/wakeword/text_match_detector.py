"""
Text-match wake word detector — legacy fallback.

Detects the wake word by checking if the STT transcript starts with
or contains the wake word string.  This is the approach used in the
original ``speech_input.py``.

Drawback: Requires running the full STT engine on every utterance,
which is expensive.  OpenWakeWord is strongly preferred for production.
This detector exists only as a fallback when OpenWakeWord is unavailable.
"""

from __future__ import annotations

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.wakeword.base import BaseWakeWordDetector, WakeWordResult

logger = get_logger("wakeword.text_match")


class TextMatchDetector(BaseWakeWordDetector):
    """Text-matching wake word detector (legacy fallback).

    This detector does NOT process audio frames directly.  Instead, it
    provides a ``check_transcript()`` method that is called after STT
    to check whether the transcribed text contains the wake word.

    The ``process_frame()`` method always returns ``detected=False``
    because this detector cannot operate on raw audio.

    Args:
        wake_word: The wake word to match (case-insensitive).
    """

    def __init__(self, wake_word: str = "simon"):
        self._wake_word = wake_word.lower()
        logger.info(f"Text-match detector initialised for '{self._wake_word}'")

    @property
    def name(self) -> str:
        return "text_match"

    @property
    def wake_word(self) -> str:
        return self._wake_word

    def process_frame(self, frame: AudioFrame) -> WakeWordResult:
        """Always returns not detected (text-match cannot process raw audio)."""
        return WakeWordResult(detected=False, confidence=0.0)

    def check_transcript(self, text: str) -> WakeWordResult:
        """Check if a transcript contains the wake word.

        Args:
            text: The transcribed text from STT.

        Returns:
            WakeWordResult with detection status.
        """
        text_lower = text.lower().strip()

        if text_lower.startswith(self._wake_word):
            return WakeWordResult(
                detected=True,
                confidence=1.0,
                keyword=self._wake_word,
            )
        elif self._wake_word in text_lower:
            return WakeWordResult(
                detected=True,
                confidence=0.8,
                keyword=self._wake_word,
            )
        return WakeWordResult(detected=False, confidence=0.0)

    def strip_wake_word(self, text: str) -> str:
        """Remove the wake word prefix from a transcript.

        Args:
            text: The transcribed text.

        Returns:
            Text with the wake word prefix removed and stripped.
        """
        text_lower = text.lower().strip()
        if text_lower.startswith(self._wake_word):
            stripped = text[len(self._wake_word):].strip()
            # Remove common separators after wake word
            for sep in [",", ".", "!", "?"]:
                stripped = stripped.lstrip(sep).strip()
            return stripped
        return text

    def reset(self) -> None:
        pass
