"""
Acoustic scene type enumeration.

Used by the Scene Classifier to categorise the current acoustic environment,
and by the Adaptive Controller to select DSP / STT parameters tuned for
each scene type.
"""

from __future__ import annotations

from enum import Enum, auto


class SceneType(Enum):
    """Acoustic scene classification labels.

    Each label maps to a parameter profile in ``config/defaults/scenes.yaml``
    that specifies optimal DSP, VAD, and STT settings for that environment.
    """

    QUIET_ROOM = auto()
    """Low ambient noise, minimal reverb.  Offices, bedrooms, libraries."""

    OFFICE = auto()
    """Moderate background noise (keyboard, AC), some speech from others."""

    CLASSROOM = auto()
    """Moderate reverb, intermittent loud speech, paper rustling."""

    HOME = auto()
    """Variable noise (TV, kitchen, family), low-to-moderate."""

    STREET = auto()
    """Traffic noise, wind, pedestrians, variable loudness."""

    BUS_TRAIN = auto()
    """Engine drone, announcements, rumble, moderate-to-high noise."""

    SHOPPING_MALL = auto()
    """Reverberant, crowd babble, music, announcements."""

    VERY_NOISY = auto()
    """Construction, concerts, heavy traffic.  SNR below 5 dB."""

    UNKNOWN = auto()
    """Scene could not be classified with sufficient confidence."""

    @property
    def is_noisy(self) -> bool:
        """True if this scene is expected to have significant background noise."""
        return self in {
            SceneType.STREET,
            SceneType.BUS_TRAIN,
            SceneType.SHOPPING_MALL,
            SceneType.VERY_NOISY,
        }

    @property
    def is_reverberant(self) -> bool:
        """True if this scene is expected to have significant reverberation."""
        return self in {
            SceneType.CLASSROOM,
            SceneType.SHOPPING_MALL,
        }

    @property
    def label(self) -> str:
        """Human-readable label for logging and display."""
        return self.name.replace("_", " ").title()
