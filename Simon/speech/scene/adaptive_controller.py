"""
Adaptive Controller — adjusts DSP, VAD, wake word, and STT parameters
based on the current acoustic scene profile.

Receives a ``SceneProfile`` from the SceneClassifier and configures
downstream components to operate optimally for that environment. For
example, a noisy street triggers higher beam sizes in STT, lower VAD
speech thresholds, and more aggressive noise suppression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from speech.models.decoding_params import DecodingParams
from speech.models.scene_type import SceneType
from speech.scene.scene_profile import SceneProfile

logger = logging.getLogger(__name__)


@dataclass
class SceneAdaptation:
    """The full set of adapted parameters for a given scene.

    Attributes:
        decoding_params: STT decoding parameters tuned for the scene.
        vad_speech_threshold: Adjusted VAD speech probability threshold.
        wakeword_threshold: Adjusted wake word detection threshold.
        noise_suppression_level: 0.0–1.0, where 1.0 = maximum suppression.
        agc_target_rms: Adjusted AGC target RMS.
        confidence_threshold_boost: Added to base confidence thresholds.
    """

    decoding_params: DecodingParams
    vad_speech_threshold: float = 0.5
    wakeword_threshold: float = 0.7
    noise_suppression_level: float = 0.5
    agc_target_rms: float = 3000.0
    confidence_threshold_boost: float = 0.0


# Pre-defined adaptation profiles per scene type
_SCENE_ADAPTATIONS: dict[SceneType, SceneAdaptation] = {
    SceneType.QUIET_ROOM: SceneAdaptation(
        decoding_params=DecodingParams.for_quiet(),
        vad_speech_threshold=0.45,
        wakeword_threshold=0.65,
        noise_suppression_level=0.2,
        agc_target_rms=2500.0,
        confidence_threshold_boost=0.0,
    ),
    SceneType.OFFICE: SceneAdaptation(
        decoding_params=DecodingParams.for_moderate_noise(),
        vad_speech_threshold=0.50,
        wakeword_threshold=0.70,
        noise_suppression_level=0.4,
        agc_target_rms=3000.0,
        confidence_threshold_boost=0.0,
    ),
    SceneType.CLASSROOM: SceneAdaptation(
        decoding_params=DecodingParams(
            beam_size=5, patience=1.2, best_of=3,
            no_speech_threshold=0.5, scene_type=SceneType.CLASSROOM,
        ),
        vad_speech_threshold=0.50,
        wakeword_threshold=0.70,
        noise_suppression_level=0.4,
        agc_target_rms=3000.0,
        confidence_threshold_boost=0.0,
    ),
    SceneType.HOME: SceneAdaptation(
        decoding_params=DecodingParams(
            beam_size=5, patience=1.0, best_of=1,
            no_speech_threshold=0.5, scene_type=SceneType.HOME,
        ),
        vad_speech_threshold=0.48,
        wakeword_threshold=0.68,
        noise_suppression_level=0.3,
        agc_target_rms=2800.0,
        confidence_threshold_boost=0.0,
    ),
    SceneType.STREET: SceneAdaptation(
        decoding_params=DecodingParams.for_noisy(),
        vad_speech_threshold=0.60,
        wakeword_threshold=0.80,
        noise_suppression_level=0.7,
        agc_target_rms=4000.0,
        confidence_threshold_boost=0.05,
    ),
    SceneType.BUS_TRAIN: SceneAdaptation(
        decoding_params=DecodingParams(
            beam_size=8, patience=1.5, best_of=5,
            no_speech_threshold=0.3, log_prob_threshold=-1.5,
            scene_type=SceneType.BUS_TRAIN,
        ),
        vad_speech_threshold=0.60,
        wakeword_threshold=0.80,
        noise_suppression_level=0.8,
        agc_target_rms=4500.0,
        confidence_threshold_boost=0.05,
    ),
    SceneType.SHOPPING_MALL: SceneAdaptation(
        decoding_params=DecodingParams(
            beam_size=8, patience=1.5, best_of=5,
            no_speech_threshold=0.4,
            scene_type=SceneType.SHOPPING_MALL,
        ),
        vad_speech_threshold=0.55,
        wakeword_threshold=0.78,
        noise_suppression_level=0.6,
        agc_target_rms=3500.0,
        confidence_threshold_boost=0.05,
    ),
    SceneType.VERY_NOISY: SceneAdaptation(
        decoding_params=DecodingParams.for_very_noisy(),
        vad_speech_threshold=0.65,
        wakeword_threshold=0.85,
        noise_suppression_level=1.0,
        agc_target_rms=5000.0,
        confidence_threshold_boost=0.10,
    ),
}

_DEFAULT_ADAPTATION = SceneAdaptation(
    decoding_params=DecodingParams(),
    vad_speech_threshold=0.50,
    wakeword_threshold=0.70,
    noise_suppression_level=0.5,
    agc_target_rms=3000.0,
    confidence_threshold_boost=0.0,
)


class AdaptiveController:
    """Receives SceneProfiles and produces SceneAdaptations.

    The controller maps classified scenes to pre-defined parameter
    profiles, ensuring each component operates optimally for the
    current acoustic environment.
    """

    def __init__(self):
        self._current_profile: Optional[SceneProfile] = None
        self._current_adaptation: SceneAdaptation = _DEFAULT_ADAPTATION
        self._last_scene: SceneType = SceneType.UNKNOWN

    @property
    def current_profile(self) -> Optional[SceneProfile]:
        return self._current_profile

    @property
    def current_adaptation(self) -> SceneAdaptation:
        return self._current_adaptation

    @property
    def current_decoding_params(self) -> DecodingParams:
        return self._current_adaptation.decoding_params

    def update(self, profile: SceneProfile) -> SceneAdaptation:
        """Update the controller with a new scene profile.

        Only triggers a parameter change if the scene type actually changed
        or the profile is significantly different.

        Args:
            profile: Latest SceneProfile from the classifier.

        Returns:
            The SceneAdaptation for the given profile.
        """
        self._current_profile = profile

        # Only log and switch if the scene actually changed
        if profile.scene_type != self._last_scene:
            logger.info(
                f"Scene changed: {self._last_scene.name} → {profile.scene_type.name} "
                f"(confidence={profile.confidence:.2f}, "
                f"SNR={profile.estimated_snr_db:.1f}dB)"
            )
            self._last_scene = profile.scene_type

        # Look up the adaptation profile
        if profile.is_reliable:
            self._current_adaptation = _SCENE_ADAPTATIONS.get(
                profile.scene_type, _DEFAULT_ADAPTATION
            )
        else:
            # Low-confidence classification: use a conservative default
            self._current_adaptation = _DEFAULT_ADAPTATION
            logger.debug(
                f"Scene confidence too low ({profile.confidence:.2f}), "
                "using default adaptation"
            )

        return self._current_adaptation

    def get_adaptation_for_scene(self, scene_type: SceneType) -> SceneAdaptation:
        """Get the adaptation profile for a specific scene type (no side effects)."""
        return _SCENE_ADAPTATIONS.get(scene_type, _DEFAULT_ADAPTATION)

    def reset(self) -> None:
        """Reset the controller to defaults."""
        self._current_profile = None
        self._current_adaptation = _DEFAULT_ADAPTATION
        self._last_scene = SceneType.UNKNOWN
