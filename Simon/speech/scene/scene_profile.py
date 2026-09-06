"""
Scene profile — rich descriptor of the current acoustic environment.

Produced by the SceneClassifier and consumed by the AdaptiveController
to tune DSP, VAD, and STT parameters in real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from speech.models.scene_type import SceneType


@dataclass
class SceneProfile:
    """Characterisation of the current acoustic environment.

    Attributes:
        scene_type:     Classified scene label.
        confidence:     Classifier confidence [0.0, 1.0] for the chosen label.
        estimated_snr_db: Estimated Signal-to-Noise Ratio in dB.
                          Positive → signal dominates; negative → noise dominates.
        reverb_estimate:  Rough reverb severity [0.0 = dry, 1.0 = very reverberant].
        ambient_rms:      RMS energy of the background noise (for logging/trending).
        timestamp:        Monotonic timestamp when this profile was generated.
    """

    scene_type: SceneType = SceneType.UNKNOWN
    confidence: float = 0.0
    estimated_snr_db: float = 30.0
    reverb_estimate: float = 0.0
    ambient_rms: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def is_noisy(self) -> bool:
        """True if SNR suggests substantial background noise."""
        return self.estimated_snr_db < 15.0

    @property
    def is_very_noisy(self) -> bool:
        """True if SNR is extremely low."""
        return self.estimated_snr_db < 5.0

    @property
    def is_reliable(self) -> bool:
        """True if the classification confidence is above a usable threshold."""
        return self.confidence >= 0.5

    def __repr__(self) -> str:
        return (
            f"SceneProfile(scene={self.scene_type.name}, "
            f"conf={self.confidence:.2f}, snr={self.estimated_snr_db:.1f}dB, "
            f"reverb={self.reverb_estimate:.2f})"
        )
