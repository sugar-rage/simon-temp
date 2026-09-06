"""
Unit tests for the EngineOrchestrator and EngineRegistry.

Tests verify:
- Registry: add/remove engines, priority ordering, health tracking
- Orchestrator: primary engine returns high confidence → no fallback
- Orchestrator: low confidence → redecoding → fallback
- Orchestrator: primary engine failure → fallback
- Orchestrator: no engines → empty transcript
- Health tracking: engine disabled after consecutive failures
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.stt.base import BaseSTTEngine
from speech.stt.engine_orchestrator import EngineOrchestrator
from speech.stt.engine_registry import EngineHealth, EngineRegistry
from speech.stt.intelligent_redecoder import IntelligentRedecoder
from speech.stt.transcript import Transcript


# ---------------------------------------------------------------------- #
#  Mock engine
# ---------------------------------------------------------------------- #

class _MockEngine(BaseSTTEngine):
    """Mock STT engine for testing."""

    def __init__(
        self,
        engine_name: str,
        transcript_text: str = "test",
        confidence: float = 0.90,
        should_fail: bool = False,
    ):
        self._name = engine_name
        self._text = transcript_text
        self._confidence = confidence
        self._loaded = False
        self._should_fail = should_fail

    def transcribe(self, audio: AudioFrame, params: Optional[DecodingParams] = None) -> Transcript:
        if self._should_fail:
            raise RuntimeError(f"Engine {self._name} crashed")
        return Transcript(
            text=self._text,
            raw_text=self._text,
            confidence=self._confidence,
            engine_name=self._name,
        )

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
    return AudioFrame(data=np.zeros(16000, dtype=np.int16), sample_rate=16000)


# ---------------------------------------------------------------------- #
#  EngineRegistry tests
# ---------------------------------------------------------------------- #

class TestEngineRegistry:
    """Tests for EngineRegistry."""

    def test_add_and_get_primary(self):
        reg = EngineRegistry()
        engine = _MockEngine("whisper")
        reg.add_engine(engine)

        assert reg.get_primary() is engine
        assert reg.engine_count == 1

    def test_priority_ordering(self):
        reg = EngineRegistry()
        e1 = _MockEngine("primary")
        e2 = _MockEngine("fallback")
        reg.add_engine(e1)
        reg.add_engine(e2)

        assert reg.get_primary() is e1
        fallbacks = reg.get_fallbacks()
        assert len(fallbacks) == 1
        assert fallbacks[0] is e2

    def test_remove_engine(self):
        reg = EngineRegistry()
        e1 = _MockEngine("primary")
        reg.add_engine(e1)
        reg.remove_engine("primary")

        assert reg.get_primary() is None
        assert reg.engine_count == 0

    def test_health_tracking_success(self):
        reg = EngineRegistry()
        engine = _MockEngine("test")
        reg.add_engine(engine)

        health = reg.get_health("test")
        assert health is not None
        health.record_success()
        assert health.total_transcriptions == 1
        assert health.consecutive_failures == 0

    def test_health_tracking_disables_after_failures(self):
        reg = EngineRegistry()
        engine = _MockEngine("test")
        reg.add_engine(engine)

        health = reg.get_health("test")
        for _ in range(5):
            health.record_failure(max_consecutive=5)

        assert health.disabled is True
        assert not health.is_available

    def test_disabled_engine_skipped_as_primary(self):
        reg = EngineRegistry()
        e1 = _MockEngine("primary")
        e2 = _MockEngine("fallback")
        reg.add_engine(e1)
        reg.add_engine(e2)

        # Disable primary
        health = reg.get_health("primary")
        for _ in range(5):
            health.record_failure(max_consecutive=5)

        # Fallback becomes the new primary
        assert reg.get_primary() is e2

    def test_re_enable_engine(self):
        reg = EngineRegistry()
        engine = _MockEngine("test")
        reg.add_engine(engine)

        health = reg.get_health("test")
        for _ in range(5):
            health.record_failure(max_consecutive=5)
        assert health.disabled is True

        health.re_enable()
        assert health.disabled is False
        assert health.is_available

    def test_get_all_available(self):
        reg = EngineRegistry()
        e1 = _MockEngine("a")
        e2 = _MockEngine("b")
        e3 = _MockEngine("c")
        reg.add_engine(e1)
        reg.add_engine(e2)
        reg.add_engine(e3)

        assert reg.available_count == 3

        # Disable one
        reg.get_health("b").disabled = True
        assert reg.available_count == 2
        assert len(reg.get_all_available()) == 2


# ---------------------------------------------------------------------- #
#  EngineOrchestrator tests
# ---------------------------------------------------------------------- #

class TestEngineOrchestrator:
    """Tests for EngineOrchestrator."""

    def test_high_confidence_no_fallback(self):
        """Primary engine returning high confidence → no fallback used."""
        reg = EngineRegistry()
        primary = _MockEngine("whisper", "navigate to library", confidence=0.95)
        primary.load()
        reg.add_engine(primary)

        orchestrator = EngineOrchestrator(reg)
        result = orchestrator.transcribe(_make_audio())

        assert result.text == "navigate to library"
        assert result.confidence == 0.95

    def test_low_confidence_triggers_fallback(self):
        """Primary engine low confidence → fallback engine tried."""
        reg = EngineRegistry()

        # Primary always returns low confidence
        primary = _MockEngine("whisper", "navigate to livery", confidence=0.30)
        primary.load()
        reg.add_engine(primary)

        # Fallback returns high confidence
        fallback = _MockEngine("vosk", "navigate to library", confidence=0.80)
        reg.add_engine(fallback)

        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
            retry_temperatures=[],  # Skip retries for this test
        )

        orchestrator = EngineOrchestrator(
            reg, redecoder=redecoder,
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
        )
        result = orchestrator.transcribe(_make_audio())

        # Should use fallback's result
        assert result.text == "navigate to library"
        assert result.engine_name == "vosk"

    def test_primary_engine_crash_triggers_fallback(self):
        """If primary engine throws, fall back to secondary."""
        reg = EngineRegistry()

        primary = _MockEngine("whisper", should_fail=True)
        primary.load()
        reg.add_engine(primary)

        fallback = _MockEngine("vosk", "read text", confidence=0.75)
        reg.add_engine(fallback)

        orchestrator = EngineOrchestrator(reg)
        result = orchestrator.transcribe(_make_audio())

        assert result.text == "read text"
        assert result.engine_name == "vosk"

    def test_no_engines_returns_empty_transcript(self):
        """With no engines registered, return empty transcript."""
        reg = EngineRegistry()
        orchestrator = EngineOrchestrator(reg)

        result = orchestrator.transcribe(_make_audio())

        assert result.text == ""

    def test_all_engines_fail_returns_empty_transcript(self):
        """If all engines fail, return empty transcript."""
        reg = EngineRegistry()
        e1 = _MockEngine("a", should_fail=True)
        e2 = _MockEngine("b", should_fail=True)
        e1.load()
        reg.add_engine(e1)
        reg.add_engine(e2)

        orchestrator = EngineOrchestrator(reg)
        result = orchestrator.transcribe(_make_audio())

        assert result.text == ""

    def test_primary_lazy_loads(self):
        """Orchestrator should load the primary engine if not loaded."""
        reg = EngineRegistry()
        primary = _MockEngine("whisper", "hello", confidence=0.95)
        # Do NOT call primary.load() — orchestrator should do it
        reg.add_engine(primary)

        orchestrator = EngineOrchestrator(reg)
        result = orchestrator.transcribe(_make_audio())

        assert primary.is_loaded
        assert result.text == "hello"

    def test_fallback_lazy_loads(self):
        """Fallback engines are lazy-loaded only when needed."""
        reg = EngineRegistry()

        primary = _MockEngine("whisper", "x", confidence=0.30)
        primary.load()
        reg.add_engine(primary)

        fallback = _MockEngine("vosk", "navigate", confidence=0.75)
        reg.add_engine(fallback)
        assert not fallback.is_loaded  # Not loaded yet

        redecoder = IntelligentRedecoder(
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
            retry_temperatures=[],
        )
        orchestrator = EngineOrchestrator(reg, redecoder=redecoder)
        orchestrator.transcribe(_make_audio())

        assert fallback.is_loaded  # Now loaded because fallback was needed

    def test_registry_property(self):
        reg = EngineRegistry()
        orchestrator = EngineOrchestrator(reg)
        assert orchestrator.registry is reg

    def test_health_recorded_on_success(self):
        reg = EngineRegistry()
        primary = _MockEngine("whisper", "ok", confidence=0.95)
        primary.load()
        reg.add_engine(primary)

        orchestrator = EngineOrchestrator(reg)
        orchestrator.transcribe(_make_audio())

        health = reg.get_health("whisper")
        assert health.total_transcriptions == 1

    def test_health_recorded_on_failure(self):
        reg = EngineRegistry()
        primary = _MockEngine("whisper", should_fail=True)
        primary.load()
        reg.add_engine(primary)

        # Add a fallback so orchestrator doesn't return empty immediately
        fallback = _MockEngine("vosk", "ok", confidence=0.9)
        reg.add_engine(fallback)

        orchestrator = EngineOrchestrator(reg)
        orchestrator.transcribe(_make_audio())

        health = reg.get_health("whisper")
        assert health.total_failures == 1
