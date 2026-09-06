"""
Configurable DSP pipeline — chain-of-responsibility audio processing.

Orchestrates the sequence: Noise Suppression → Echo Cancellation → AGC.
Each stage implements ``process(AudioFrame) -> AudioFrame`` and can be
individually enabled/disabled, reordered, or replaced at runtime.

Design decision: Chain-of-responsibility pattern rather than a monolithic
process function.  This allows:
1. Stages to be independently tested
2. The Adaptive Controller to reconfigure the pipeline per scene
3. New stages to be added without modifying existing code (OCP)
4. Each stage to report its own latency to the metrics system
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector

logger = get_logger("audio.dsp_pipeline")
metrics = get_collector()


class DSPStage(ABC):
    """Abstract base class for a DSP pipeline stage.

    Every stage must implement ``process()`` and provide a ``name``.
    Stages can be enabled/disabled at runtime.
    """

    def __init__(self, name: str, enabled: bool = True):
        self._name = name
        self._enabled = enabled

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info(f"DSP stage '{self._name}' {'enabled' if value else 'disabled'}")

    @abstractmethod
    def process(self, frame: AudioFrame) -> AudioFrame:
        """Process an audio frame and return the result."""
        ...


class NoiseSuppressorStage(DSPStage):
    """DSP stage wrapping the NoiseSuppressor."""

    def __init__(self, suppressor=None, enabled: bool = True):
        super().__init__("noise_suppressor", enabled)
        self._suppressor = suppressor

    def process(self, frame: AudioFrame) -> AudioFrame:
        if self._suppressor is None:
            return frame
        return self._suppressor.process(frame)


class GainControllerStage(DSPStage):
    """DSP stage wrapping the GainController."""

    def __init__(self, controller=None, enabled: bool = True):
        super().__init__("gain_controller", enabled)
        self._controller = controller

    def process(self, frame: AudioFrame) -> AudioFrame:
        if self._controller is None:
            return frame
        return self._controller.process(frame)


class EchoCancellerStage(DSPStage):
    """DSP stage for echo cancellation (placeholder for Phase 3)."""

    def __init__(self, enabled: bool = False):
        super().__init__("echo_canceller", enabled)

    def process(self, frame: AudioFrame) -> AudioFrame:
        # Echo cancellation will be implemented in Phase 3.
        # For now, pass through unchanged.
        return frame


class DSPPipeline:
    """Configurable audio processing pipeline.

    Stages are executed in order.  Each stage that is enabled will process
    the audio frame, reporting its latency to the metrics system.

    Args:
        stages: Ordered list of DSP stages.  If None, a default pipeline
                is created.
    """

    def __init__(self, stages: list[DSPStage] | None = None):
        self._stages: list[DSPStage] = stages or []

    @property
    def stages(self) -> list[DSPStage]:
        """The ordered list of pipeline stages."""
        return list(self._stages)

    @property
    def stage_names(self) -> list[str]:
        """Names of all stages in order."""
        return [s.name for s in self._stages]

    def add_stage(self, stage: DSPStage, index: int | None = None) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: The DSP stage to add.
            index: Position to insert at (None = append to end).
        """
        if index is None:
            self._stages.append(stage)
        else:
            self._stages.insert(index, stage)
        logger.info(f"Added DSP stage '{stage.name}' at position {index or len(self._stages) - 1}")

    def remove_stage(self, name: str) -> Optional[DSPStage]:
        """Remove a stage by name.

        Returns:
            The removed stage, or None if not found.
        """
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                removed = self._stages.pop(i)
                logger.info(f"Removed DSP stage '{name}'")
                return removed
        return None

    def get_stage(self, name: str) -> Optional[DSPStage]:
        """Get a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                return stage
        return None

    def enable_stage(self, name: str) -> None:
        """Enable a stage by name."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = True

    def disable_stage(self, name: str) -> None:
        """Disable a stage by name."""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = False

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Run the audio frame through all enabled stages.

        Args:
            frame: Input AudioFrame.

        Returns:
            Processed AudioFrame.
        """
        result = frame
        for stage in self._stages:
            if not stage.enabled:
                continue

            t0 = time.monotonic()
            try:
                result = stage.process(result)
            except Exception as e:
                logger.error(
                    f"DSP stage '{stage.name}' failed: {e}",
                    extra={"stage": stage.name},
                )
                metrics.counter(f"dsp.{stage.name}.errors")
                # Continue with the unprocessed frame rather than crashing
                continue

            elapsed_ms = (time.monotonic() - t0) * 1000
            metrics.histogram(f"dsp.{stage.name}.latency_ms", elapsed_ms)

        return result

    @classmethod
    def create_default(
        cls,
        noise_suppressor=None,
        gain_controller=None,
        enable_noise_suppression: bool = True,
        enable_agc: bool = True,
        enable_echo_cancellation: bool = False,
    ) -> DSPPipeline:
        """Create the default DSP pipeline: Noise → Echo Cancel → AGC.

        Args:
            noise_suppressor: NoiseSuppressor instance (or None for passthrough).
            gain_controller:  GainController instance (or None for passthrough).
            enable_noise_suppression: Whether to enable the noise suppressor stage.
            enable_agc: Whether to enable the AGC stage.
            enable_echo_cancellation: Whether to enable echo cancellation.
        """
        stages = [
            NoiseSuppressorStage(noise_suppressor, enabled=enable_noise_suppression),
            EchoCancellerStage(enabled=enable_echo_cancellation),
            GainControllerStage(gain_controller, enabled=enable_agc),
        ]
        pipeline = cls(stages)
        logger.info(
            f"DSP pipeline created: {[s.name for s in stages if s.enabled]}"
        )
        return pipeline
