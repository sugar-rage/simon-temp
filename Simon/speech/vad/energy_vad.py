"""
Energy-based VAD fallback — simple RMS energy thresholding.

Used when Silero VAD cannot be loaded (no PyTorch, model download failure).
Much lower accuracy than Silero but zero dependencies beyond NumPy.
Suitable as a last-resort fallback to keep the system operational.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.vad.base import BaseVAD, VADResult

logger = get_logger("vad.energy")


class EnergyVAD(BaseVAD):
    """RMS energy-based Voice Activity Detection.

    Classifies a frame as speech if its RMS energy exceeds a threshold.

    Args:
        energy_threshold:        RMS threshold for speech detection (int16 scale).
        min_speech_duration_ms:  Minimum speech segment duration.
        min_silence_duration_ms: Silence needed to end a segment.
        max_speech_duration_s:   Maximum segment length before forced end.
    """

    def __init__(
        self,
        energy_threshold: float = 300.0,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 700,
        max_speech_duration_s: float = 30.0,
    ):
        self._threshold = energy_threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._max_speech_s = max_speech_duration_s

        self._is_speaking = False
        self._speech_frames: list[AudioFrame] = []
        self._silence_counter_ms: float = 0.0
        self._speech_counter_ms: float = 0.0
        self._completed_segment: Optional[AudioFrame] = None

        # Adaptive noise floor estimation
        self._noise_floor: float = 100.0
        self._noise_alpha: float = 0.95

        logger.info(f"Energy VAD initialised (threshold={energy_threshold})")

    @property
    def name(self) -> str:
        return "energy_vad"

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def set_threshold(self, threshold: float) -> None:
        """Update energy threshold (from adaptive controller)."""
        self._threshold = max(50.0, threshold)

    def process_frame(self, frame: AudioFrame) -> VADResult:
        """Process frame using RMS energy thresholding."""
        rms = frame.rms_energy
        frame_ms = frame.duration_s * 1000

        # Update noise floor estimate (slow adaptation during silence)
        if not self._is_speaking:
            self._noise_floor = (
                self._noise_alpha * self._noise_floor
                + (1 - self._noise_alpha) * rms
            )

        # Speech detection: RMS must exceed noise floor + threshold
        effective_threshold = max(self._threshold, self._noise_floor * 2.0)
        is_speech = rms > effective_threshold
        confidence = min(1.0, rms / (effective_threshold * 2)) if effective_threshold > 0 else 0.0

        speech_start = False
        speech_end = False

        if is_speech:
            self._silence_counter_ms = 0.0
            if not self._is_speaking:
                self._is_speaking = True
                speech_start = True
                self._speech_counter_ms = 0.0

            self._speech_frames.append(frame)
            self._speech_counter_ms += frame_ms

            if self._speech_counter_ms >= self._max_speech_s * 1000:
                speech_end = True
                self._emit_segment()
        else:
            if self._is_speaking:
                self._silence_counter_ms += frame_ms
                self._speech_frames.append(frame)

                if self._silence_counter_ms >= self._min_silence_ms:
                    speech_end = True
                    self._emit_segment()

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            speech_start=speech_start,
            speech_end=speech_end,
        )

    def _emit_segment(self) -> None:
        """Package accumulated frames into a speech segment."""
        if not self._speech_frames:
            self._is_speaking = False
            return

        total_ms = sum(f.duration_s * 1000 for f in self._speech_frames)
        if total_ms < self._min_speech_ms:
            self._speech_frames.clear()
            self._is_speaking = False
            return

        try:
            self._completed_segment = AudioFrame.concatenate(self._speech_frames)
        except Exception:
            pass

        self._speech_frames.clear()
        self._is_speaking = False

    def get_speech_segment(self) -> Optional[AudioFrame]:
        segment = self._completed_segment
        self._completed_segment = None
        return segment

    def reset(self) -> None:
        self._is_speaking = False
        self._speech_frames.clear()
        self._silence_counter_ms = 0.0
        self._speech_counter_ms = 0.0
        self._completed_segment = None
