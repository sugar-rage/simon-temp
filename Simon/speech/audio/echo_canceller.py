"""
Echo Canceller — acoustic echo cancellation for the SIMON speech subsystem.

When SIMON speaks through the device's speakers, the microphone picks up
the output audio as echo.  The echo canceller removes this echo so the
STT engine doesn't recognize SIMON's own speech as user commands.

Uses a simple spectral subtraction approach:
1. When TTS output starts, record the reference signal
2. When microphone input arrives, subtract the estimated echo
3. Apply spectral gating to suppress residual echo

This is a lightweight fallback.  For production, consider using
system-level AEC (e.g. WebRTC AEC, PulseAudio echo cancellation).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EchoCanceller:
    """Simple spectral subtraction echo canceller.

    Args:
        sample_rate:     Audio sample rate.
        frame_duration_ms: Frame duration in milliseconds.
        echo_decay:      Echo energy decay factor (0.0–1.0).
        buffer_seconds:  Seconds of reference audio to keep.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        echo_decay: float = 0.7,
        buffer_seconds: float = 2.0,
    ):
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate * frame_duration_ms / 1000)
        self._echo_decay = echo_decay
        self._buffer_frames = int(buffer_seconds * 1000 / frame_duration_ms)
        self._reference: deque[np.ndarray] = deque(maxlen=self._buffer_frames)
        self._is_playing = False
        self._lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        """True if TTS output is currently active."""
        return self._is_playing

    def on_playback_start(self) -> None:
        """Notify the echo canceller that TTS playback has started."""
        with self._lock:
            self._is_playing = True
            self._reference.clear()
        logger.debug("Echo canceller: playback started")

    def on_playback_stop(self) -> None:
        """Notify the echo canceller that TTS playback has stopped."""
        with self._lock:
            self._is_playing = False
        logger.debug("Echo canceller: playback stopped")

    def feed_reference(self, audio: np.ndarray) -> None:
        """Feed the TTS output signal as echo reference.

        Call this for every audio chunk that is being played through speakers.

        Args:
            audio: Output audio data (float32).
        """
        with self._lock:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            self._reference.append(audio.copy())

    def cancel(self, microphone_audio: np.ndarray) -> np.ndarray:
        """Remove echo from microphone input.

        Args:
            microphone_audio: Raw microphone input (float32).

        Returns:
            Echo-cancelled audio.
        """
        with self._lock:
            if not self._is_playing or not self._reference:
                return microphone_audio

            mic = microphone_audio.astype(np.float32) if microphone_audio.dtype != np.float32 else microphone_audio.copy()

            # Estimate echo energy from reference buffer
            if self._reference:
                ref_concat = np.concatenate(list(self._reference))
                ref_energy = np.sqrt(np.mean(ref_concat ** 2))
            else:
                return mic

            if ref_energy < 1e-6:
                return mic

            # Simple energy-based gating
            mic_energy = np.sqrt(np.mean(mic ** 2))

            if mic_energy < ref_energy * self._echo_decay:
                # Microphone signal is likely dominated by echo — suppress
                suppression = 1.0 - (ref_energy * self._echo_decay / max(mic_energy, 1e-10))
                suppression = max(suppression, 0.05)  # Don't completely zero out
                mic *= suppression
                logger.debug(
                    f"Echo suppressed: mic_energy={mic_energy:.4f}, "
                    f"ref_energy={ref_energy:.4f}, suppression={suppression:.3f}"
                )

            return mic

    def reset(self) -> None:
        """Reset the echo canceller state."""
        with self._lock:
            self._is_playing = False
            self._reference.clear()
