"""
Automatic Gain Controller (AGC) — normalises audio levels.

Ensures that quiet speech is amplified and loud speech is attenuated to
a consistent target level.  This is critical for ASR accuracy because
Whisper-family models are sensitive to input amplitude.

Design decision: Peak-normalised AGC with smoothed gain rather than
compression-based AGC.  Peak normalisation is simpler, introduces less
distortion, and is well-suited for speech (short utterances with
predictable dynamic range).
"""

from __future__ import annotations

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger

logger = get_logger("audio.agc")


class GainController:
    """Automatic Gain Controller with smoothed peak normalisation.

    Args:
        target_rms:      Target RMS energy level (int16 scale).
        min_gain:        Minimum gain multiplier (prevents over-attenuation).
        max_gain:        Maximum gain multiplier (prevents over-amplification).
        attack_coeff:    Smoothing coefficient for gain increase (0-1, lower = slower).
        release_coeff:   Smoothing coefficient for gain decrease (0-1, lower = slower).
    """

    def __init__(
        self,
        target_rms: float = 3000.0,
        min_gain: float = 0.5,
        max_gain: float = 10.0,
        attack_coeff: float = 0.1,
        release_coeff: float = 0.05,
    ):
        self._target_rms = target_rms
        self._min_gain = min_gain
        self._max_gain = max_gain
        self._attack = attack_coeff
        self._release = release_coeff
        self._current_gain = 1.0

    @property
    def current_gain(self) -> float:
        """The current gain being applied."""
        return self._current_gain

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Apply automatic gain control to an audio frame.

        Args:
            frame: Input AudioFrame (any dtype).

        Returns:
            Gain-normalised AudioFrame (same dtype as input).
        """
        if frame.is_empty:
            return frame

        # Work in float64 for precision
        audio = frame.data.astype(np.float64)
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1.0:
            # Near-silence — don't adjust gain (would amplify noise)
            return frame

        # Compute desired gain
        desired_gain = self._target_rms / rms
        desired_gain = np.clip(desired_gain, self._min_gain, self._max_gain)

        # Smooth gain changes to avoid sudden volume jumps
        if desired_gain > self._current_gain:
            coeff = self._attack
        else:
            coeff = self._release

        self._current_gain = (
            coeff * desired_gain + (1 - coeff) * self._current_gain
        )

        # Apply gain
        gained = audio * self._current_gain

        # Clip and convert back to original dtype
        if frame.dtype == "int16":
            gained = np.clip(gained, -32768, 32767).astype(np.int16)
        elif frame.dtype == "float32":
            gained = np.clip(gained, -1.0, 1.0).astype(np.float32)
        else:
            gained = gained.astype(frame.data.dtype)

        return AudioFrame(
            data=gained,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            dtype=frame.dtype,
            timestamp=frame.timestamp,
            device_id=frame.device_id,
            sequence_id=frame.sequence_id,
        )

    def set_target_rms(self, target: float) -> None:
        """Update the target RMS level (for adaptive scene control)."""
        self._target_rms = target
