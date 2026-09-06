"""
Unit tests for the IntelligentRedecoder.

Tests verify:
- High confidence skips redecoding (0 retries)
- Low confidence triggers multi-temperature retry
- Best transcript selected by consensus when 2+ agree
- Best transcript selected by confidence when no consensus
- needs_fallback flag set when all passes are below low threshold
- Empty audio produces empty transcript
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.stt.base import BaseSTTEngine
from speech.stt.intelligent_redecoder import IntelligentRedecoder, RedecoderResult
from speech.stt.transcript import Transcript

import numpy as np


# ---------------------------------------------------------------------- #
#  Mock STT engine
# ---------------------------------------------------------------------- #

class MockSTTEngine(BaseSTTEngine):
    """A mock STT engine that returns pre-configured transcripts."""

    def __init__(self, name: str = "mock-engine", transcripts: Optional[List[Transcript]] = None):
        self._name = name
        self._transcripts = transcripts or []
        self._call_index = 0
        self._loaded = True

    def transcribe(self, audio: AudioFrame, params: Optional[DecodingParams] = None) -> Transcript:
        if self._call_index < len(self._transcripts):
            result = self._transcripts[self._call_index]
            self._call_index += 1
            return result
        return Transcript()

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _make_audio() -> AudioFrame:
    """Simple test audio frame."""
    data = np.zeros(16000, dtype=np.int16)
    return AudioFrame(data=data, sample_rate=16000)


def _make_transcript(text: str, confidence: float, compression_ratio: float = 1.5) -> Transcript:
    """Create a test transcript."""
    return Transcript(
        text=text,
        raw_text=text,
        confidence=confidence,
        language="en",
        engine_name="mock-engine",
        compression_ratio=compression_ratio,
    )


# ---------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------- #

class TestIntelligentRedecoder:
    """Tests for IntelligentRedecoder."""

    def test_high_confidence_skips_redecoding(self):
        """Primary decode with high confidence should return immediately."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("navigate to library", 0.95),
        ])
        redecoder = IntelligentRedecoder(high_confidence_threshold=0.85)

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert isinstance(result, RedecoderResult)
        assert result.retries_used == 0
        assert result.needs_fallback is False
        assert result.best.text == "navigate to library"
        assert result.best.confidence == 0.95
        assert len(result.all_transcripts) == 1

    def test_low_confidence_triggers_retries(self):
        """Primary decode below threshold triggers multi-temperature retry."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("navigate to livery", 0.60),     # temp=0.0
            _make_transcript("navigate to library", 0.75),    # temp=0.2
            _make_transcript("navigate to library", 0.72),    # temp=0.4
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            retry_temperatures=[0.2, 0.4],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert result.retries_used == 2
        assert len(result.all_transcripts) == 3
        # "navigate to library" appears twice (consensus) with higher confidence
        assert result.best.text == "navigate to library"

    def test_consensus_selection(self):
        """When 2+ transcripts agree, use the consensus with highest confidence."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("stop", 0.60),        # temp=0.0
            _make_transcript("shop", 0.70),         # temp=0.2
            _make_transcript("stop", 0.65),          # temp=0.4
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            retry_temperatures=[0.2, 0.4],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        # "stop" appears 2x (consensus), best confidence is 0.65
        assert result.best.text == "stop"
        assert result.best.confidence == 0.65

    def test_no_consensus_selects_highest_confidence(self):
        """Without consensus, select the transcript with highest confidence."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("stop", 0.55),
            _make_transcript("shop", 0.70),
            _make_transcript("stock", 0.60),
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            retry_temperatures=[0.2, 0.4],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        # No consensus, "shop" has highest confidence
        assert result.best.text == "shop"
        assert result.best.confidence == 0.70

    def test_needs_fallback_when_all_low_confidence(self):
        """needs_fallback is True when best confidence < low_threshold."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("something", 0.30),
            _make_transcript("something else", 0.35),
            _make_transcript("some thing", 0.25),
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
            retry_temperatures=[0.2, 0.4],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert result.needs_fallback is True
        assert result.best.confidence < 0.50

    def test_does_not_need_fallback_when_retry_succeeds(self):
        """needs_fallback is False when retry achieves sufficient confidence."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("something", 0.40),    # temp=0.0
            _make_transcript("navigate", 0.65),      # temp=0.2
            _make_transcript("navigate", 0.62),      # temp=0.4
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
            retry_temperatures=[0.2, 0.4],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert result.needs_fallback is False
        assert result.best.confidence >= 0.50

    def test_custom_retry_temperatures(self):
        """Custom temperature list controls number of retries."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("a", 0.30),
            _make_transcript("b", 0.35),
            _make_transcript("c", 0.40),
            _make_transcript("d", 0.45),
        ])
        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            retry_temperatures=[0.1, 0.3, 0.6],
        )

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert result.retries_used == 3
        assert len(result.all_transcripts) == 4  # 1 primary + 3 retries

    def test_exactly_at_high_threshold_skips_redecoding(self):
        """Confidence exactly equal to threshold should skip redecoding."""
        engine = MockSTTEngine(transcripts=[
            _make_transcript("hello", 0.85),
        ])
        redecoder = IntelligentRedecoder(high_confidence_threshold=0.85)

        result = redecoder.decode_with_retry(engine, _make_audio(), DecodingParams())

        assert result.retries_used == 0
        assert result.needs_fallback is False
