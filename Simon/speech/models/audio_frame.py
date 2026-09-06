"""
Audio frame data model — the universal currency between all pipeline stages.

Every component in the DSP pipeline, VAD, wake word detector, and STT engine
operates on AudioFrame instances.  This ensures type safety, prevents
sample-rate mismatches, and carries metadata (timestamps, device source)
through the entire processing chain.

Design decision: Immutable-by-convention dataclass rather than a frozen
dataclass, because NumPy arrays are inherently mutable.  We use __slots__
for memory efficiency since thousands of frames flow through the pipeline
per second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AudioFrame:
    """A single chunk of audio with associated metadata.

    Attributes:
        data:        Raw audio samples as a 1-D NumPy array.
        sample_rate: Sample rate in Hz (e.g. 16000).
        channels:    Number of audio channels (1 = mono).
        dtype:       NumPy dtype string of ``data`` (e.g. 'int16', 'float32').
        timestamp:   Monotonic timestamp (seconds) when this frame was captured.
        duration_s:  Duration of the frame in seconds (computed from data length).
        device_id:   ID of the audio device that produced this frame, or None.
        sequence_id: Monotonically increasing frame counter (for ordering).
    """

    data: np.ndarray
    sample_rate: int
    channels: int = 1
    dtype: str = "int16"
    timestamp: float = field(default_factory=time.monotonic)
    device_id: Optional[int] = None
    sequence_id: int = 0

    # ------------------------------------------------------------------ #
    #  Computed properties
    # ------------------------------------------------------------------ #

    @property
    def duration_s(self) -> float:
        """Duration of this frame in seconds."""
        if self.sample_rate == 0:
            return 0.0
        return len(self.data) / (self.sample_rate * self.channels)

    @property
    def num_samples(self) -> int:
        """Total number of samples (across all channels)."""
        return len(self.data)

    @property
    def is_mono(self) -> bool:
        return self.channels == 1

    @property
    def is_empty(self) -> bool:
        return self.data is None or len(self.data) == 0

    @property
    def rms_energy(self) -> float:
        """Root-mean-square energy of the audio signal.

        Useful for quick silence detection without a full VAD model.
        """
        if self.is_empty:
            return 0.0
        audio_float = self.data.astype(np.float64)
        return float(np.sqrt(np.mean(audio_float ** 2)))

    @property
    def peak_amplitude(self) -> float:
        """Peak absolute amplitude in the frame."""
        if self.is_empty:
            return 0.0
        return float(np.max(np.abs(self.data.astype(np.float64))))

    # ------------------------------------------------------------------ #
    #  Type conversion helpers
    # ------------------------------------------------------------------ #

    def to_float32(self) -> AudioFrame:
        """Return a new AudioFrame with data normalised to float32 [-1.0, 1.0].

        This is the format expected by most ML models (Whisper, Silero VAD,
        DeepFilterNet).  Int16 samples are divided by 32768.0.
        """
        if self.dtype == "float32":
            return self

        if self.dtype == "int16":
            converted = self.data.astype(np.float32) / 32768.0
        elif self.dtype == "float64":
            converted = self.data.astype(np.float32)
        else:
            # Best-effort: attempt direct cast
            converted = self.data.astype(np.float32)

        return AudioFrame(
            data=converted,
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            timestamp=self.timestamp,
            device_id=self.device_id,
            sequence_id=self.sequence_id,
        )

    def to_int16(self) -> AudioFrame:
        """Return a new AudioFrame with data as int16 [-32768, 32767].

        This is the format expected by sounddevice recording and webrtcvad.
        Float32 samples are multiplied by 32768 and clipped.
        """
        if self.dtype == "int16":
            return self

        if self.dtype == "float32" or self.dtype == "float64":
            clipped = np.clip(self.data, -1.0, 1.0)
            converted = (clipped * 32768.0).astype(np.int16)
        else:
            converted = self.data.astype(np.int16)

        return AudioFrame(
            data=converted,
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            timestamp=self.timestamp,
            device_id=self.device_id,
            sequence_id=self.sequence_id,
        )

    def to_mono(self) -> AudioFrame:
        """Down-mix multi-channel audio to mono by averaging channels."""
        if self.channels == 1:
            return self

        # Reshape to (num_frames, channels), then average
        reshaped = self.data.reshape(-1, self.channels)
        mono = reshaped.mean(axis=1)

        if self.dtype == "int16":
            mono = mono.astype(np.int16)
        else:
            mono = mono.astype(np.float32)

        return AudioFrame(
            data=mono,
            sample_rate=self.sample_rate,
            channels=1,
            dtype=self.dtype,
            timestamp=self.timestamp,
            device_id=self.device_id,
            sequence_id=self.sequence_id,
        )

    def resample(self, target_rate: int) -> AudioFrame:
        """Resample to *target_rate* Hz using polyphase filtering.

        Uses ``scipy.signal.resample_poly`` for high-quality resampling.
        Falls back to linear interpolation if scipy is unavailable.
        """
        if self.sample_rate == target_rate:
            return self

        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(self.sample_rate, target_rate)
            up = target_rate // g
            down = self.sample_rate // g
            resampled = resample_poly(self.data, up, down).astype(
                self.data.dtype
            )
        except ImportError:
            # Fallback: simple linear interpolation
            num_target = int(len(self.data) * target_rate / self.sample_rate)
            indices = np.linspace(0, len(self.data) - 1, num_target)
            resampled = np.interp(indices, np.arange(len(self.data)), self.data)
            resampled = resampled.astype(self.data.dtype)

        return AudioFrame(
            data=resampled,
            sample_rate=target_rate,
            channels=self.channels,
            dtype=self.dtype,
            timestamp=self.timestamp,
            device_id=self.device_id,
            sequence_id=self.sequence_id,
        )

    # ------------------------------------------------------------------ #
    #  Concatenation
    # ------------------------------------------------------------------ #

    @staticmethod
    def concatenate(frames: list[AudioFrame]) -> AudioFrame:
        """Concatenate a list of AudioFrames into a single frame.

        All frames must share the same sample rate, channels, and dtype.

        Raises:
            ValueError: If frames have mismatched properties.
        """
        if not frames:
            raise ValueError("Cannot concatenate empty list of frames")

        ref = frames[0]
        for f in frames[1:]:
            if f.sample_rate != ref.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: {ref.sample_rate} vs {f.sample_rate}"
                )
            if f.channels != ref.channels:
                raise ValueError(
                    f"Channel mismatch: {ref.channels} vs {f.channels}"
                )

        combined = np.concatenate([f.data for f in frames])
        return AudioFrame(
            data=combined,
            sample_rate=ref.sample_rate,
            channels=ref.channels,
            dtype=ref.dtype,
            timestamp=frames[0].timestamp,
            device_id=ref.device_id,
            sequence_id=frames[0].sequence_id,
        )

    # ------------------------------------------------------------------ #
    #  Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"AudioFrame(samples={self.num_samples}, "
            f"rate={self.sample_rate}Hz, "
            f"ch={self.channels}, "
            f"dtype={self.dtype}, "
            f"dur={self.duration_s:.3f}s, "
            f"rms={self.rms_energy:.1f})"
        )
