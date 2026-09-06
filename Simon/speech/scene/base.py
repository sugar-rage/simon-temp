"""
Abstract base interface for acoustic scene classifiers.

All scene classifiers (ONNX model, heuristic energy, etc.) implement
``BaseSceneClassifier`` so the AdaptiveController can swap them
transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from speech.models.audio_frame import AudioFrame
from speech.scene.scene_profile import SceneProfile


class BaseSceneClassifier(ABC):
    """Abstract base class for acoustic scene classifiers."""

    @abstractmethod
    def classify(self, audio: AudioFrame) -> SceneProfile:
        """Classify an audio frame and return a SceneProfile.

        Args:
            audio: A chunk of audio (typically 1-2 seconds) to classify.

        Returns:
            A ``SceneProfile`` describing the detected acoustic environment.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. smoothing buffers)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this classifier."""
        ...
