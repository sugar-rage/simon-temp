"""
Unit tests for Voice Activity Detection.

Tests the EnergyVAD (which requires no ML dependencies) and the VAD
base interface contract.  SileroVAD tests are skipped if torch is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech.models.audio_frame import AudioFrame
from speech.tests.conftest import make_noise_frame, make_silence_frame, make_sine_frame
from speech.vad.base import BaseVAD, VADResult
from speech.vad.energy_vad import EnergyVAD


class TestVADResult:
    """Tests for the VADResult data class."""

    def test_default_values(self):
        r = VADResult()
        assert r.is_speech is False
        assert r.confidence == 0.0
        assert r.speech_start is False
        assert r.speech_end is False

    def test_repr(self):
        r = VADResult(is_speech=True, confidence=0.85)
        assert "speech=True" in repr(r)
        assert "0.85" in repr(r)


class TestEnergyVAD:
    """Tests for the RMS energy-based VAD."""

    def test_silence_not_detected_as_speech(self):
        """Silent audio should not trigger speech detection."""
        vad = EnergyVAD(energy_threshold=200)
        frame = make_silence_frame(duration_s=0.03)
        result = vad.process_frame(frame)
        assert result.is_speech is False

    def test_loud_signal_detected_as_speech(self):
        """A loud sine wave should trigger speech detection."""
        vad = EnergyVAD(energy_threshold=100)
        frame = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        result = vad.process_frame(frame)
        assert result.is_speech is True

    def test_speech_start_transition(self):
        """First speech frame should set speech_start=True."""
        vad = EnergyVAD(energy_threshold=100)
        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        result = vad.process_frame(speech)
        assert result.speech_start is True

    def test_speech_end_after_silence(self):
        """Speech should end after sufficient silence frames."""
        vad = EnergyVAD(
            energy_threshold=100,
            min_silence_duration_ms=60,
            min_speech_duration_ms=30,
        )

        # Send speech frames
        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        vad.process_frame(speech)  # speech_start
        vad.process_frame(speech)  # continuing speech

        # Send enough silence to end the segment
        silence = make_silence_frame(duration_s=0.03)
        end_result = None
        for _ in range(10):  # 10 × 30ms = 300ms > min_silence_duration_ms
            result = vad.process_frame(silence)
            if result.speech_end:
                end_result = result
                break

        assert end_result is not None
        assert end_result.speech_end is True

    def test_get_speech_segment_returns_accumulated_audio(self):
        """After speech_end, get_speech_segment should return the audio."""
        vad = EnergyVAD(
            energy_threshold=100,
            min_silence_duration_ms=60,
            min_speech_duration_ms=30,
        )

        # Feed speech
        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        vad.process_frame(speech)
        vad.process_frame(speech)

        # Feed silence to trigger end
        silence = make_silence_frame(duration_s=0.03)
        for _ in range(10):
            vad.process_frame(silence)

        segment = vad.get_speech_segment()
        assert segment is not None
        assert segment.num_samples > 0
        assert segment.duration_s > 0

    def test_get_speech_segment_clears_after_read(self):
        """get_speech_segment should return None on second call."""
        vad = EnergyVAD(
            energy_threshold=100,
            min_silence_duration_ms=60,
            min_speech_duration_ms=30,
        )

        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        vad.process_frame(speech)
        vad.process_frame(speech)

        silence = make_silence_frame(duration_s=0.03)
        for _ in range(10):
            vad.process_frame(silence)

        first = vad.get_speech_segment()
        second = vad.get_speech_segment()
        assert first is not None
        assert second is None

    def test_short_speech_discarded(self):
        """Speech shorter than min_speech_duration_ms should be discarded."""
        vad = EnergyVAD(
            energy_threshold=100,
            min_silence_duration_ms=60,
            min_speech_duration_ms=200,  # 200ms minimum
        )

        # One short frame of speech (30ms < 200ms)
        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        vad.process_frame(speech)

        silence = make_silence_frame(duration_s=0.03)
        for _ in range(10):
            vad.process_frame(silence)

        segment = vad.get_speech_segment()
        assert segment is None  # Too short, should be discarded

    def test_reset_clears_state(self):
        vad = EnergyVAD(energy_threshold=100)
        speech = make_sine_frame(frequency=300, amplitude=0.6, duration_s=0.03)
        vad.process_frame(speech)
        assert vad.is_speaking is True

        vad.reset()
        assert vad.is_speaking is False
        assert vad.get_speech_segment() is None

    def test_set_threshold(self):
        vad = EnergyVAD(energy_threshold=200)
        vad.set_threshold(500)
        assert vad._threshold == 500

    def test_set_threshold_minimum_bound(self):
        vad = EnergyVAD()
        vad.set_threshold(10)
        assert vad._threshold == 50.0  # Minimum is 50

    def test_name_property(self):
        vad = EnergyVAD()
        assert vad.name == "energy_vad"


class TestSileroVAD:
    """Tests for Silero VAD (skipped if torch not available)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch required for Silero VAD tests")

    def test_name_property(self):
        from speech.vad.silero_vad import SileroVAD
        vad = SileroVAD()
        assert vad.name == "silero_vad"
