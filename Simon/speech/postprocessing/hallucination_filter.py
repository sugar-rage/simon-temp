"""
Whisper hallucination filter — detects and rejects common hallucination patterns.

Whisper-family models are known to hallucinate in several characteristic ways:
1. Repetitive text ("the the the the the...")
2. Phantom speech in silence ("Thank you for watching", "Subscribe to my channel")
3. High compression ratio with low log probability
4. Music/noise transcribed as unrelated text

This filter catches these patterns before they reach intent parsing,
preventing false command execution from hallucinated text.
"""

from __future__ import annotations

import re
from typing import Optional

from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.stt.transcript import Transcript

logger = get_logger("postprocessing.hallucination_filter")
metrics = get_collector()

# Common Whisper hallucination phrases (case-insensitive)
_HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe to",
    "please subscribe",
    "like and subscribe",
    "thank you for listening",
    "see you next time",
    "see you in the next",
    "bye bye",
    "goodbye",
    "music playing",
    "applause",
    "laughter",
    "cheering",
    "silence",
    "no audio",
    "foreign language",
    "inaudible",
    "unintelligible",
    "background noise",
    "♪",
    "♫",
    "subtitles by",
    "captions by",
    "translated by",
    "transcribed by",
    "copyright",
    "all rights reserved",
    "you",
]

# Patterns that indicate repetitive hallucination
_REPETITION_PATTERN = re.compile(r"(\b\w+\b)(?:\s+\1){3,}", re.IGNORECASE)

# Pattern for text that's just punctuation or whitespace
_EMPTY_PATTERN = re.compile(r"^[\s\.\,\!\?\;\:\-\—\–\"\']*$")


class HallucinationFilter:
    """Detects and rejects Whisper hallucination patterns.

    Args:
        max_compression_ratio:  Reject if compression ratio exceeds this.
        min_log_prob:           Reject if avg log prob is below this.
        max_no_speech_prob:     Reject if no-speech probability exceeds this.
        max_repetition_ratio:   Reject if repeated words exceed this ratio.
        enable_phrase_filter:   Whether to check against known hallucination phrases.
    """

    def __init__(
        self,
        max_compression_ratio: float = 2.4,
        min_log_prob: float = -1.0,
        max_no_speech_prob: float = 0.7,
        max_repetition_ratio: float = 0.5,
        enable_phrase_filter: bool = True,
    ):
        self._max_compression = max_compression_ratio
        self._min_log_prob = min_log_prob
        self._max_no_speech = max_no_speech_prob
        self._max_repetition = max_repetition_ratio
        self._phrase_filter = enable_phrase_filter

    def check(self, transcript: Transcript) -> FilterResult:
        """Check a transcript for hallucination patterns.

        Args:
            transcript: The STT transcript to evaluate.

        Returns:
            ``FilterResult`` indicating whether the transcript is valid.
        """
        text = transcript.text.strip()

        # Empty or whitespace-only
        if not text or _EMPTY_PATTERN.match(text):
            return FilterResult(
                is_valid=False,
                reason="Empty or whitespace-only transcript",
            )

        # Single character (unlikely to be a valid command)
        if len(text) <= 1:
            return FilterResult(is_valid=False, reason="Single character transcript")

        # Compression ratio check
        if (
            transcript.compression_ratio > self._max_compression
            and transcript.avg_log_prob < self._min_log_prob
        ):
            metrics.counter("hallucination.compression_ratio")
            return FilterResult(
                is_valid=False,
                reason=(
                    f"High compression ratio ({transcript.compression_ratio:.1f}) "
                    f"with low log prob ({transcript.avg_log_prob:.2f})"
                ),
            )

        # No-speech probability check
        if transcript.no_speech_prob > self._max_no_speech and text:
            metrics.counter("hallucination.no_speech_prob")
            return FilterResult(
                is_valid=False,
                reason=f"High no-speech probability ({transcript.no_speech_prob:.2f})",
            )

        # Repetition check
        words = text.split()
        if len(words) >= 4:
            if _REPETITION_PATTERN.search(text):
                metrics.counter("hallucination.repetition")
                return FilterResult(
                    is_valid=False,
                    reason="Repetitive text detected",
                )

            # Check ratio of unique words to total words
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < (1 - self._max_repetition) and len(words) > 6:
                metrics.counter("hallucination.low_unique_ratio")
                return FilterResult(
                    is_valid=False,
                    reason=f"Low word diversity ({unique_ratio:.2f})",
                )

        # Known hallucination phrase check
        if self._phrase_filter:
            text_lower = text.lower()
            for phrase in _HALLUCINATION_PHRASES:
                if phrase in text_lower:
                    # Only block if the hallucination phrase IS the entire text
                    # (not if a command happens to contain a common word)
                    if len(text_lower) - len(phrase) < 10:
                        metrics.counter("hallucination.known_phrase")
                        return FilterResult(
                            is_valid=False,
                            reason=f"Known hallucination phrase: '{phrase}'",
                        )

        return FilterResult(is_valid=True)

    def filter(self, transcript: Transcript) -> Optional[Transcript]:
        """Filter a transcript, returning it if valid or None if hallucination.

        Convenience wrapper around ``check()``.
        """
        result = self.check(transcript)
        if not result.is_valid:
            logger.warning(
                f"Hallucination filtered",
                extra={
                    "text": transcript.text[:50],
                    "reason": result.reason,
                    "confidence": f"{transcript.confidence:.2f}",
                },
            )
            metrics.counter("hallucination.total_filtered")
            return None
        return transcript


class FilterResult:
    """Result of a hallucination filter check.

    Attributes:
        is_valid: True if the transcript is NOT a hallucination.
        reason:   Explanation if the transcript was flagged.
    """

    __slots__ = ("is_valid", "reason")

    def __init__(self, is_valid: bool, reason: str = ""):
        self.is_valid = is_valid
        self.reason = reason

    def __repr__(self) -> str:
        return f"FilterResult(valid={self.is_valid}, reason={self.reason!r})"
