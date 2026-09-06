"""
Silero VAD implementation — state-of-the-art voice activity detection.

Silero VAD is a compact neural-network-based VAD that provides much higher
accuracy than energy-based or webrtcvad approaches.  It correctly handles
whispered speech, non-speech vocalizations, and noisy environments.

This module:
- Loads the Silero VAD model from the PyTorch Hub (cached locally)
- Processes 30ms audio frames and returns speech probability
- Tracks speech segment boundaries (start / end)
- Accumulates speech frames into complete utterance segments
- Supports adaptive threshold from the scene classifier

Design decision: We accumulate speech frames into a buffer and emit
complete segments when silence is detected for ``min_silence_duration_ms``.
This is more robust than emitting per-frame results because it handles
brief pauses within speech (e.g. "navigate... to... library").
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.vad.base import BaseVAD, VADResult

logger = get_logger("vad.silero")
metrics = get_collector()


class SileroVAD(BaseVAD):
    """Silero VAD implementation with speech segment accumulation.

    Args:
        speech_threshold:        Probability above which a frame is considered speech.
        min_speech_duration_ms:  Minimum speech duration to emit a segment.
        min_silence_duration_ms: Silence duration required to end a segment.
        max_speech_duration_s:   Maximum segment length before forced emission.
        padding_duration_ms:     Audio padding added before/after speech boundaries.
        sample_rate:             Expected audio sample rate (must be 16000 for Silero).
    """

    def __init__(
        self,
        speech_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 700,
        max_speech_duration_s: float = 30.0,
        padding_duration_ms: int = 300,
        sample_rate: int = 16000,
    ):
        self._threshold = speech_threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._max_speech_s = max_speech_duration_s
        self._padding_ms = padding_duration_ms
        self._sample_rate = sample_rate

        # Silero model
        self._model = None
        self._model_loaded = False

        # State machine
        self._is_speaking = False
        self._speech_frames: list[AudioFrame] = []
        self._padding_frames: list[AudioFrame] = []
        self._silence_counter_ms: float = 0.0
        self._speech_counter_ms: float = 0.0
        self._completed_segment: Optional[AudioFrame] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Silero VAD model from torch.hub."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._model_loaded = True
            logger.info("Silero VAD model loaded successfully")
        except ImportError:
            logger.error("PyTorch not installed — Silero VAD unavailable")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")

    @property
    def name(self) -> str:
        return "silero_vad"

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def set_threshold(self, threshold: float) -> None:
        """Update speech detection threshold (for adaptive scene control)."""
        self._threshold = max(0.1, min(0.95, threshold))
        logger.debug(f"VAD threshold updated to {self._threshold:.2f}")

    def process_frame(self, frame: AudioFrame) -> VADResult:
        """Process a single audio frame through Silero VAD.

        Args:
            frame: AudioFrame at 16kHz, mono.

        Returns:
            VADResult with speech probability and segment boundary flags.
        """
        if not self._model_loaded:
            # Fallback: treat everything as speech if model not loaded
            return VADResult(is_speech=True, confidence=0.5)

        # Silero expects float32 at 16kHz
        f32 = frame.to_float32()
        if f32.sample_rate != 16000:
            f32 = f32.resample(16000)

        # Run inference
        try:
            import torch
            audio_tensor = torch.from_numpy(f32.data)
            confidence = float(self._model(audio_tensor, self._sample_rate).item())
        except Exception as e:
            logger.warning(f"Silero VAD inference failed: {e}")
            metrics.counter("vad.inference_errors")
            return VADResult(is_speech=False, confidence=0.0)

        is_speech = confidence >= self._threshold
        frame_duration_ms = frame.duration_s * 1000

        # Record metrics
        metrics.gauge("vad.speech_probability", confidence)

        # State machine
        speech_start = False
        speech_end = False

        if is_speech:
            self._silence_counter_ms = 0.0

            if not self._is_speaking:
                # Transition: silence → speech
                self._is_speaking = True
                speech_start = True
                self._speech_counter_ms = 0.0
                # Include padding frames (audio just before speech started)
                self._speech_frames = list(self._padding_frames)
                self._padding_frames.clear()
                logger.debug("Speech started", extra={"confidence": f"{confidence:.2f}"})

            # Accumulate speech frame
            self._speech_frames.append(frame)
            self._speech_counter_ms += frame_duration_ms

            # Force-emit if max duration exceeded
            if self._speech_counter_ms >= self._max_speech_s * 1000:
                speech_end = True
                self._emit_segment()
                logger.warning("Speech segment force-ended (max duration)")

        else:
            # Maintain padding buffer (circular, for pre-speech context)
            max_padding_frames = int(self._padding_ms / frame_duration_ms) if frame_duration_ms > 0 else 10
            self._padding_frames.append(frame)
            if len(self._padding_frames) > max_padding_frames:
                self._padding_frames.pop(0)

            if self._is_speaking:
                self._silence_counter_ms += frame_duration_ms
                # Still accumulate frames during silence (for trailing context)
                self._speech_frames.append(frame)

                if self._silence_counter_ms >= self._min_silence_ms:
                    # Transition: speech → silence
                    speech_end = True
                    self._emit_segment()

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            speech_start=speech_start,
            speech_end=speech_end,
        )

    def _emit_segment(self) -> None:
        """Package accumulated speech frames into a complete segment."""
        if not self._speech_frames:
            self._is_speaking = False
            return

        # Check minimum speech duration
        total_duration_ms = sum(f.duration_s * 1000 for f in self._speech_frames)
        if total_duration_ms < self._min_speech_ms:
            logger.debug(
                f"Speech segment too short ({total_duration_ms:.0f}ms < "
                f"{self._min_speech_ms}ms), discarding"
            )
            self._speech_frames.clear()
            self._is_speaking = False
            metrics.counter("vad.segments_too_short")
            return

        try:
            segment = AudioFrame.concatenate(self._speech_frames)
            self._completed_segment = segment
            metrics.histogram("vad.speech_duration_ms", total_duration_ms)
            logger.debug(
                f"Speech segment emitted",
                extra={"duration_ms": f"{total_duration_ms:.0f}"},
            )
        except Exception as e:
            logger.error(f"Failed to concatenate speech frames: {e}")

        self._speech_frames.clear()
        self._is_speaking = False

    def get_speech_segment(self) -> Optional[AudioFrame]:
        """Retrieve the last completed speech segment.

        Returns the segment and clears it.  Returns None if no
        complete segment is available.
        """
        segment = self._completed_segment
        self._completed_segment = None
        return segment

    def reset(self) -> None:
        """Reset all internal state."""
        self._is_speaking = False
        self._speech_frames.clear()
        self._padding_frames.clear()
        self._silence_counter_ms = 0.0
        self._speech_counter_ms = 0.0
        self._completed_segment = None
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass
