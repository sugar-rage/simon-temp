"""
Unit tests for the hallucination filter.

Tests detection of common Whisper hallucination patterns:
repetition, known phrases, high compression ratio, no-speech probability.
"""

from __future__ import annotations

import pytest

from speech.postprocessing.hallucination_filter import HallucinationFilter
from speech.stt.transcript import Transcript


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _make_transcript(
    text: str,
    confidence: float = 0.5,
    avg_log_prob: float = -0.5,
    compression_ratio: float = 1.5,
    no_speech_prob: float = 0.1,
) -> Transcript:
    return Transcript(
        text=text,
        raw_text=text,
        confidence=confidence,
        avg_log_prob=avg_log_prob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
        engine_name="test",
    )


# ---------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------- #

class TestHallucinationFilter:
    """Tests for the HallucinationFilter."""

    def test_valid_command_passes(self):
        f = HallucinationFilter()
        t = _make_transcript("navigate to the library")
        result = f.check(t)
        assert result.is_valid is True

    def test_empty_transcript_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("")
        result = f.check(t)
        assert result.is_valid is False

    def test_whitespace_only_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("   ...  ")
        result = f.check(t)
        assert result.is_valid is False

    def test_single_char_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("a")
        result = f.check(t)
        assert result.is_valid is False

    def test_repetitive_text_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("the the the the the the the")
        result = f.check(t)
        assert result.is_valid is False

    def test_known_hallucination_phrase_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("thank you for watching")
        result = f.check(t)
        assert result.is_valid is False

    def test_known_phrase_in_longer_text_allowed(self):
        """A known phrase embedded in a real command should NOT be filtered."""
        f = HallucinationFilter()
        t = _make_transcript(
            "navigate to the coffee shop thank you for telling me"
        )
        result = f.check(t)
        # The phrase is embedded, not the whole text
        assert result.is_valid is True

    def test_high_compression_with_low_logprob_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript(
            "something something",
            compression_ratio=3.0,
            avg_log_prob=-1.5,
        )
        result = f.check(t)
        assert result.is_valid is False

    def test_high_compression_with_good_logprob_passes(self):
        """High compression alone (without low logprob) should not reject."""
        f = HallucinationFilter()
        t = _make_transcript(
            "something something",
            compression_ratio=3.0,
            avg_log_prob=-0.3,  # Good logprob
        )
        result = f.check(t)
        assert result.is_valid is True

    def test_high_no_speech_prob_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript(
            "some phantom text",
            no_speech_prob=0.85,
        )
        result = f.check(t)
        assert result.is_valid is False

    def test_low_unique_ratio_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("go go go go go go go go")
        result = f.check(t)
        assert result.is_valid is False

    def test_filter_returns_none_for_hallucination(self):
        f = HallucinationFilter()
        t = _make_transcript("thank you for watching")
        assert f.filter(t) is None

    def test_filter_returns_transcript_for_valid(self):
        f = HallucinationFilter()
        t = _make_transcript("navigate to VIT")
        assert f.filter(t) is t

    def test_music_symbols_rejected(self):
        f = HallucinationFilter()
        t = _make_transcript("♪ ♫")
        result = f.check(t)
        assert result.is_valid is False

    def test_phrase_filter_disabled(self):
        """When phrase filter is off, known phrases should pass."""
        f = HallucinationFilter(enable_phrase_filter=False)
        t = _make_transcript("thank you for watching")
        result = f.check(t)
        # Other checks (compression, no_speech) may still catch it,
        # but the phrase check itself should be skipped
        # With default params, it should pass since other metrics are fine
        assert result.is_valid is True
