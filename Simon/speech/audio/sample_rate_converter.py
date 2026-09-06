"""
Sample rate converter — high-quality polyphase resampling.

Centralises all resampling logic so that individual components don't need
to handle rate conversion internally.  Uses ``scipy.signal.resample_poly``
for quality, with a linear interpolation fallback.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger

logger = get_logger("audio.resampler")


class SampleRateConverter:
    """Resamples audio frames between sample rates.

    Args:
        target_rate: Target sample rate in Hz.
    """

    def __init__(self, target_rate: int = 16000):
        self._target_rate = target_rate
        self._has_scipy = False
        try:
            from scipy.signal import resample_poly  # noqa: F401
            self._has_scipy = True
        except ImportError:
            logger.warning("scipy not available — using linear interpolation fallback")

    @property
    def target_rate(self) -> int:
        return self._target_rate

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Resample an AudioFrame to the target rate.

        If the frame is already at the target rate, returns it unchanged
        (zero-copy).
        """
        if frame.sample_rate == self._target_rate:
            return frame
        return frame.resample(self._target_rate)
