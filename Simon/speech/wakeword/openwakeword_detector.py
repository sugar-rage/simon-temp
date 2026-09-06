"""
OpenWakeWord detector — dedicated neural wake word detection.

OpenWakeWord provides always-on, low-CPU wake word detection using
a small CNN model.  Unlike text-matching (which requires running the
full STT engine on every utterance), OpenWakeWord operates directly on
audio frames and consumes minimal resources.

This module:
- Loads the OpenWakeWord model for the configured keyword
- Processes 30ms audio frames and returns detection probability
- Enforces a cooldown period to prevent rapid re-triggering
- Supports adaptive threshold from the scene classifier
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.wakeword.base import BaseWakeWordDetector, WakeWordResult

logger = get_logger("wakeword.openwakeword")
metrics = get_collector()


class OpenWakeWordDetector(BaseWakeWordDetector):
    """OpenWakeWord-based wake word detector.

    Args:
        wake_word:    The wake word to detect (e.g. "simon").
        threshold:    Detection confidence threshold [0.0, 1.0].
        model_path:   Path to a custom model file, or None for default.
        cooldown_s:   Minimum seconds between consecutive activations.
    """

    def __init__(
        self,
        wake_word: str = "simon",
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        cooldown_s: float = 1.0,
    ):
        self._wake_word = wake_word.lower()
        self._threshold = threshold
        self._model_path = model_path
        self._cooldown_s = cooldown_s

        self._model = None
        self._model_loaded = False
        self._last_detection_time: float = 0.0

        self._load_model()

    def _load_model(self) -> None:
        """Load the OpenWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model

            if self._model_path:
                self._model = Model(wakeword_models=[self._model_path])
            else:
                self._model = Model()

            self._model_loaded = True
            logger.info(
                f"OpenWakeWord loaded for '{self._wake_word}'",
                extra={"threshold": self._threshold},
            )
        except ImportError:
            logger.warning("openwakeword not installed (pip install openwakeword)")
        except Exception as e:
            logger.error(f"Failed to load OpenWakeWord: {e}")

    @property
    def name(self) -> str:
        return "openwakeword"

    @property
    def wake_word(self) -> str:
        return self._wake_word

    def set_threshold(self, threshold: float) -> None:
        """Update detection threshold (for adaptive scene control)."""
        self._threshold = max(0.3, min(0.95, threshold))

    def process_frame(self, frame: AudioFrame) -> WakeWordResult:
        """Check for wake word in an audio frame.

        Args:
            frame: 16kHz mono AudioFrame.

        Returns:
            WakeWordResult with detection status.
        """
        if not self._model_loaded:
            return WakeWordResult(detected=False, confidence=0.0)

        # Cooldown check
        now = time.monotonic()
        if now - self._last_detection_time < self._cooldown_s:
            return WakeWordResult(detected=False, confidence=0.0)

        try:
            # OpenWakeWord expects int16 numpy array
            i16 = frame.to_int16()
            prediction = self._model.predict(i16.data)

            # Find the best matching model score
            best_score = 0.0
            best_keyword = ""
            for model_name, score in prediction.items():
                model_lower = model_name.lower().replace("_", " ")
                if self._wake_word in model_lower or model_lower in self._wake_word:
                    if score > best_score:
                        best_score = score
                        best_keyword = model_name

            # Also check all scores if wake word not directly found
            if best_score == 0.0:
                for model_name, score in prediction.items():
                    if score > best_score:
                        best_score = score
                        best_keyword = model_name

            detected = best_score >= self._threshold

            if detected:
                self._last_detection_time = now
                metrics.counter("wakeword.detections")
                logger.info(
                    f"Wake word detected: '{best_keyword}'",
                    extra={"confidence": f"{best_score:.2f}"},
                )

            return WakeWordResult(
                detected=detected,
                confidence=best_score,
                keyword=best_keyword,
            )

        except Exception as e:
            logger.warning(f"OpenWakeWord inference error: {e}")
            metrics.counter("wakeword.inference_errors")
            return WakeWordResult(detected=False, confidence=0.0)

    def reset(self) -> None:
        """Reset internal state."""
        self._last_detection_time = 0.0
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass
