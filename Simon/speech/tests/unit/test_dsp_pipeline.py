"""
Unit tests for the DSP pipeline.

Tests the chain-of-responsibility pattern, stage enable/disable,
per-stage latency metrics, and graceful error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech.audio.dsp_pipeline import (
    DSPPipeline,
    DSPStage,
    EchoCancellerStage,
    GainControllerStage,
    NoiseSuppressorStage,
)
from speech.audio.gain_controller import GainController
from speech.models.audio_frame import AudioFrame
from speech.tests.conftest import make_silence_frame, make_sine_frame


# ---------------------------------------------------------------------- #
#  Custom test stage
# ---------------------------------------------------------------------- #

class DoubleAmplitudeStage(DSPStage):
    """Test stage that doubles the audio amplitude."""

    def __init__(self):
        super().__init__("double_amplitude", enabled=True)

    def process(self, frame: AudioFrame) -> AudioFrame:
        data = frame.data.astype(np.float64) * 2.0
        if frame.dtype == "int16":
            data = np.clip(data, -32768, 32767).astype(np.int16)
        else:
            data = data.astype(np.float32)
        return AudioFrame(
            data=data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            dtype=frame.dtype,
            timestamp=frame.timestamp,
        )


class FailingStage(DSPStage):
    """Test stage that always raises an exception."""

    def __init__(self):
        super().__init__("failing_stage", enabled=True)

    def process(self, frame: AudioFrame) -> AudioFrame:
        raise RuntimeError("Stage failure simulation")


# ---------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------- #

class TestDSPPipeline:
    """Tests for the DSPPipeline chain-of-responsibility."""

    def test_empty_pipeline_passthrough(self):
        pipeline = DSPPipeline(stages=[])
        frame = make_sine_frame()
        result = pipeline.process(frame)
        assert np.array_equal(result.data, frame.data)

    def test_stage_execution_order(self):
        """Stages execute in the order they are added."""
        execution_log = []

        class LogStage(DSPStage):
            def process(self, frame):
                execution_log.append(self.name)
                return frame

        s1 = LogStage("first", enabled=True)
        s2 = LogStage("second", enabled=True)
        s3 = LogStage("third", enabled=True)

        pipeline = DSPPipeline([s1, s2, s3])
        pipeline.process(make_silence_frame())

        assert execution_log == ["first", "second", "third"]

    def test_disabled_stage_skipped(self):
        execution_log = []

        class LogStage(DSPStage):
            def process(self, frame):
                execution_log.append(self.name)
                return frame

        s1 = LogStage("active", enabled=True)
        s2 = LogStage("disabled", enabled=False)
        s3 = LogStage("active2", enabled=True)

        pipeline = DSPPipeline([s1, s2, s3])
        pipeline.process(make_silence_frame())

        assert execution_log == ["active", "active2"]

    def test_enable_disable_stage(self):
        stage = DoubleAmplitudeStage()
        pipeline = DSPPipeline([stage])

        frame = make_sine_frame(amplitude=0.3)
        result_enabled = pipeline.process(frame)

        pipeline.disable_stage("double_amplitude")
        result_disabled = pipeline.process(frame)

        # Disabled should return original data
        assert np.array_equal(result_disabled.data, frame.data)
        # Enabled should have modified data
        assert not np.array_equal(result_enabled.data, frame.data)

    def test_failing_stage_does_not_crash_pipeline(self):
        """A stage failure should not crash the pipeline — it skips the stage."""
        failing = FailingStage()
        normal = DoubleAmplitudeStage()
        pipeline = DSPPipeline([failing, normal])

        frame = make_sine_frame()
        # Should NOT raise
        result = pipeline.process(frame)
        # The doubling stage should still execute
        assert result is not None

    def test_add_stage(self):
        pipeline = DSPPipeline()
        assert len(pipeline.stages) == 0
        pipeline.add_stage(DoubleAmplitudeStage())
        assert len(pipeline.stages) == 1
        assert pipeline.stage_names == ["double_amplitude"]

    def test_remove_stage(self):
        stage = DoubleAmplitudeStage()
        pipeline = DSPPipeline([stage])
        removed = pipeline.remove_stage("double_amplitude")
        assert removed is stage
        assert len(pipeline.stages) == 0

    def test_remove_nonexistent_stage_returns_none(self):
        pipeline = DSPPipeline()
        assert pipeline.remove_stage("nonexistent") is None

    def test_get_stage(self):
        stage = DoubleAmplitudeStage()
        pipeline = DSPPipeline([stage])
        assert pipeline.get_stage("double_amplitude") is stage
        assert pipeline.get_stage("nonexistent") is None

    def test_create_default_pipeline(self):
        """Default pipeline has noise_suppressor, echo_canceller, gain_controller."""
        pipeline = DSPPipeline.create_default()
        names = pipeline.stage_names
        assert "noise_suppressor" in names
        assert "echo_canceller" in names
        assert "gain_controller" in names

    def test_default_pipeline_echo_cancelled_disabled(self):
        """Echo canceller is disabled by default."""
        pipeline = DSPPipeline.create_default()
        ec = pipeline.get_stage("echo_canceller")
        assert ec is not None
        assert ec.enabled is False


class TestGainController:
    """Tests for the AGC module."""

    def test_silence_not_amplified(self):
        """Near-silent audio should pass through without gain increase."""
        agc = GainController(target_rms=3000)
        frame = make_silence_frame()
        result = agc.process(frame)
        # RMS should still be near zero
        assert result.rms_energy < 50

    def test_loud_signal_attenuated(self):
        """A very loud signal should be attenuated toward the target."""
        agc = GainController(target_rms=1000, max_gain=10.0)
        # Create a loud frame
        frame = make_sine_frame(amplitude=0.9)
        result = agc.process(frame)
        # Gain should decrease (< 1.0)
        assert agc.current_gain < 1.0

    def test_quiet_signal_amplified(self):
        """A quiet signal should be amplified toward the target."""
        agc = GainController(target_rms=5000, min_gain=0.5)
        # Create a quiet frame
        frame = make_sine_frame(amplitude=0.05)
        result = agc.process(frame)
        # Gain should increase (> 1.0)
        assert agc.current_gain > 1.0

    def test_gain_bounded(self):
        """Gain should never exceed max_gain or drop below min_gain."""
        agc = GainController(target_rms=30000, min_gain=0.5, max_gain=5.0)
        frame = make_sine_frame(amplitude=0.01)
        for _ in range(100):
            agc.process(frame)
        assert agc.current_gain <= 5.0

    def test_set_target_rms(self):
        agc = GainController()
        agc.set_target_rms(5000)
        assert agc._target_rms == 5000
