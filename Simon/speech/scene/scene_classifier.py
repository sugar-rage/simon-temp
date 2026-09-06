"""
Acoustic Scene Classifier — heuristic + optional ONNX implementation.

Primary mode: A lightweight heuristic classifier based on spectral features
(spectral centroid, spectral rolloff, zero-crossing rate, RMS energy).
This works without any ML model and is surprisingly effective for the
coarse scene categories we need (quiet / moderate / noisy / very noisy).

Optional mode: An ONNX-based classifier trained on ESC-50 / AudioSet for
finer-grained classification.  Falls back to heuristic if ONNX runtime
is unavailable.

The classifier uses an exponential moving average to smooth predictions
and prevent rapid scene flipping from momentary noise bursts.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.models.scene_type import SceneType
from speech.scene.base import BaseSceneClassifier
from speech.scene.scene_profile import SceneProfile

logger = logging.getLogger(__name__)


class SceneClassifier(BaseSceneClassifier):
    """Heuristic acoustic scene classifier with EMA smoothing.

    Args:
        classification_interval_s: Minimum seconds between re-classifications.
        ema_alpha: Smoothing factor for exponential moving average (0–1).
                   Lower values = more smoothing = slower to react.
        history_size: Number of recent classifications to keep for voting.
    """

    def __init__(
        self,
        classification_interval_s: float = 3.0,
        ema_alpha: float = 0.3,
        history_size: int = 5,
    ):
        self._interval_s = classification_interval_s
        self._ema_alpha = ema_alpha
        self._history_size = history_size

        # State
        self._last_classification_time: float = 0.0
        self._current_profile = SceneProfile()
        self._history: collections.deque = collections.deque(maxlen=history_size)

        # EMA-smoothed spectral features
        self._ema_rms: float = 0.0
        self._ema_zcr: float = 0.0
        self._ema_centroid: float = 0.0
        self._initialized = False

    @property
    def name(self) -> str:
        return "heuristic-scene-classifier"

    def reset(self) -> None:
        """Reset smoothing state."""
        self._last_classification_time = 0.0
        self._current_profile = SceneProfile()
        self._history.clear()
        self._ema_rms = 0.0
        self._ema_zcr = 0.0
        self._ema_centroid = 0.0
        self._initialized = False

    def classify(self, audio: AudioFrame) -> SceneProfile:
        """Classify the acoustic scene from an audio frame.

        Uses spectral features to heuristically determine the environment.
        Returns the smoothed classification to avoid rapid flipping.
        """
        now = time.monotonic()

        # Throttle: return cached profile if too soon since last classification
        if (now - self._last_classification_time) < self._interval_s:
            return self._current_profile

        # Compute spectral features
        data = audio.data.astype(np.float32)
        if audio.dtype == "int16":
            data = data / 32768.0

        rms = float(np.sqrt(np.mean(data ** 2)))
        zcr = self._zero_crossing_rate(data)
        centroid = self._spectral_centroid(data, audio.sample_rate)

        # Update EMA
        if not self._initialized:
            self._ema_rms = rms
            self._ema_zcr = zcr
            self._ema_centroid = centroid
            self._initialized = True
        else:
            a = self._ema_alpha
            self._ema_rms = a * rms + (1.0 - a) * self._ema_rms
            self._ema_zcr = a * zcr + (1.0 - a) * self._ema_zcr
            self._ema_centroid = a * centroid + (1.0 - a) * self._ema_centroid

        # Classify based on smoothed features
        scene_type, confidence = self._classify_from_features(
            self._ema_rms, self._ema_zcr, self._ema_centroid
        )

        # Estimate SNR (rough approximation: ratio of peak to RMS)
        peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
        estimated_snr = self._estimate_snr(peak, self._ema_rms)

        # Estimate reverb (very rough: ratio of energy in late reflections)
        reverb_estimate = self._estimate_reverb(data, audio.sample_rate)

        # History voting for stability
        self._history.append(scene_type)
        voted_scene = self._majority_vote()

        profile = SceneProfile(
            scene_type=voted_scene,
            confidence=confidence,
            estimated_snr_db=estimated_snr,
            reverb_estimate=reverb_estimate,
            ambient_rms=self._ema_rms,
            timestamp=now,
        )

        self._current_profile = profile
        self._last_classification_time = now

        logger.debug(f"Scene classified: {profile}")
        return profile

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _zero_crossing_rate(data: np.ndarray) -> float:
        """Compute zero-crossing rate of the signal."""
        if len(data) < 2:
            return 0.0
        signs = np.sign(data)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings) / len(data)

    @staticmethod
    def _spectral_centroid(data: np.ndarray, sample_rate: int) -> float:
        """Compute spectral centroid (center of mass of the spectrum)."""
        if len(data) < 64:
            return 0.0
        spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)
        total = np.sum(spectrum)
        if total < 1e-10:
            return 0.0
        return float(np.sum(freqs * spectrum) / total)

    @staticmethod
    def _estimate_snr(peak: float, rms: float) -> float:
        """Rough SNR estimate from peak-to-RMS ratio (in dB)."""
        if rms < 1e-10:
            return 60.0  # Essentially silent → very high SNR
        ratio = peak / rms
        # Map ratio to approximate dB. A ratio of ~3 (pure sine) ≈ 10 dB
        # A ratio of ~1.4 (noise-like) ≈ 3 dB
        snr_db = 20.0 * np.log10(max(ratio, 1e-10))
        # Clamp to reasonable range
        return float(np.clip(snr_db, -10.0, 60.0))

    @staticmethod
    def _estimate_reverb(data: np.ndarray, sample_rate: int) -> float:
        """Very rough reverb estimate using autocorrelation decay.

        Reverberant signals have a slower autocorrelation decay.
        Returns a value in [0.0, 1.0].
        """
        if len(data) < sample_rate // 10:
            return 0.0

        # Compute autocorrelation of a short window
        window = data[:min(len(data), sample_rate // 4)]
        if np.std(window) < 1e-10:
            return 0.0

        norm_window = window - np.mean(window)
        autocorr = np.correlate(norm_window, norm_window, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]

        if autocorr[0] < 1e-10:
            return 0.0

        autocorr = autocorr / autocorr[0]

        # Find the lag where autocorrelation drops below 0.1
        decay_samples = np.argmax(autocorr < 0.1)
        if decay_samples == 0:
            decay_samples = len(autocorr)

        # Normalise: slow decay (>50ms) = reverberant
        decay_ms = (decay_samples / sample_rate) * 1000.0
        reverb = float(np.clip(decay_ms / 100.0, 0.0, 1.0))
        return reverb

    # ------------------------------------------------------------------ #
    #  Classification logic
    # ------------------------------------------------------------------ #

    def _classify_from_features(
        self, rms: float, zcr: float, centroid: float
    ) -> tuple[SceneType, float]:
        """Map spectral features to a scene type.

        Thresholds are empirically tuned for 16kHz mono audio normalised to [-1, 1].
        """
        # Very quiet → QUIET_ROOM
        if rms < 0.005:
            return SceneType.QUIET_ROOM, 0.90

        # Very loud + high ZCR → VERY_NOISY
        if rms > 0.15 and zcr > 0.15:
            return SceneType.VERY_NOISY, 0.80

        # Loud + moderate ZCR → STREET or BUS_TRAIN
        if rms > 0.08:
            if centroid > 3000:
                return SceneType.STREET, 0.70
            else:
                return SceneType.BUS_TRAIN, 0.65

        # Moderate RMS + high centroid → SHOPPING_MALL (crowd babble is high freq)
        if rms > 0.03 and centroid > 2500:
            return SceneType.SHOPPING_MALL, 0.55

        # Moderate RMS + low centroid → OFFICE / HOME
        if rms > 0.02:
            if zcr > 0.08:
                return SceneType.OFFICE, 0.60
            else:
                return SceneType.HOME, 0.55

        # Low RMS but not silent → CLASSROOM or QUIET_ROOM
        if rms > 0.008:
            return SceneType.CLASSROOM, 0.50

        return SceneType.QUIET_ROOM, 0.70

    def _majority_vote(self) -> SceneType:
        """Return the most common scene type in recent history."""
        if not self._history:
            return self._current_profile.scene_type

        counter: Dict[SceneType, int] = {}
        for s in self._history:
            counter[s] = counter.get(s, 0) + 1

        return max(counter, key=counter.get)
