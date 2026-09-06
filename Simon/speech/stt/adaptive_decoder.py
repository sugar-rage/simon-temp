"""
Adaptive Decoder — maps acoustic scene profiles to STT decoding parameters.

This module receives SceneProfiles from the SceneClassifier (via the
AdaptiveController) and produces DecodingParams tuned for the current
environment. The parameters are consumed by the EngineOrchestrator.

Unlike the hardcoded DecodingParams factory methods (for_quiet, for_noisy),
this module can load custom scene-to-parameter mappings from scenes.yaml
and can interpolate between profiles for intermediate SNR values.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from speech.models.decoding_params import DecodingParams
from speech.models.scene_type import SceneType
from speech.scene.scene_profile import SceneProfile

logger = logging.getLogger(__name__)


# Default scene-to-params mapping (can be overridden via scenes.yaml)
_DEFAULT_SCENE_PARAMS: Dict[SceneType, Dict] = {
    SceneType.QUIET_ROOM: dict(
        beam_size=3, temperature=0.0, patience=1.0, best_of=1,
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    ),
    SceneType.OFFICE: dict(
        beam_size=5, temperature=0.0, patience=1.0, best_of=3,
        no_speech_threshold=0.5, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    ),
    SceneType.CLASSROOM: dict(
        beam_size=5, temperature=0.0, patience=1.2, best_of=3,
        no_speech_threshold=0.5, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    ),
    SceneType.HOME: dict(
        beam_size=5, temperature=0.0, patience=1.0, best_of=1,
        no_speech_threshold=0.5, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    ),
    SceneType.STREET: dict(
        beam_size=8, temperature=0.0, patience=1.5, best_of=5,
        no_speech_threshold=0.4, log_prob_threshold=-1.5,
        compression_ratio_threshold=2.8,
    ),
    SceneType.BUS_TRAIN: dict(
        beam_size=8, temperature=0.0, patience=1.5, best_of=5,
        no_speech_threshold=0.3, log_prob_threshold=-1.5,
        compression_ratio_threshold=2.8,
    ),
    SceneType.SHOPPING_MALL: dict(
        beam_size=8, temperature=0.0, patience=1.5, best_of=5,
        no_speech_threshold=0.4, log_prob_threshold=-1.5,
        compression_ratio_threshold=2.8,
    ),
    SceneType.VERY_NOISY: dict(
        beam_size=10, temperature=0.0, patience=2.0, best_of=5,
        no_speech_threshold=0.3, log_prob_threshold=-2.0,
        compression_ratio_threshold=3.0,
    ),
}


class AdaptiveDecoder:
    """Produces DecodingParams tuned for the current acoustic scene.

    Args:
        scene_params: Optional custom mapping of SceneType → param dict.
                      Falls back to _DEFAULT_SCENE_PARAMS if not provided.
        base_language: Default language code.
    """

    def __init__(
        self,
        scene_params: Optional[Dict[SceneType, Dict]] = None,
        base_language: str = "en",
    ):
        self._scene_params = scene_params or _DEFAULT_SCENE_PARAMS
        self._base_language = base_language
        self._current_params = DecodingParams(language=base_language)
        self._last_scene = SceneType.UNKNOWN

    @property
    def current_params(self) -> DecodingParams:
        return self._current_params

    def get_params(
        self,
        profile: Optional[SceneProfile] = None,
        initial_prompt: Optional[str] = None,
    ) -> DecodingParams:
        """Get decoding parameters for the current scene.

        Args:
            profile: Current scene profile (if None, uses last known).
            initial_prompt: Optional vocabulary-biasing prompt.

        Returns:
            DecodingParams tuned for the scene.
        """
        scene = profile.scene_type if profile else self._last_scene

        # Look up the parameter dict for this scene
        param_dict = self._scene_params.get(scene, {})

        params = DecodingParams(
            beam_size=param_dict.get("beam_size", 5),
            temperature=param_dict.get("temperature", 0.0),
            patience=param_dict.get("patience", 1.0),
            best_of=param_dict.get("best_of", 1),
            compression_ratio_threshold=param_dict.get(
                "compression_ratio_threshold", 2.4
            ),
            no_speech_threshold=param_dict.get("no_speech_threshold", 0.6),
            log_prob_threshold=param_dict.get("log_prob_threshold", -1.0),
            initial_prompt=initial_prompt,
            language=self._base_language,
            scene_type=scene,
        )

        if scene != self._last_scene:
            logger.info(
                f"Adaptive decoding: {self._last_scene.name} → {scene.name} "
                f"(beam={params.beam_size}, patience={params.patience}, "
                f"best_of={params.best_of})"
            )
            self._last_scene = scene

        self._current_params = params
        return params

    def get_params_for_scene(self, scene_type: SceneType) -> DecodingParams:
        """Get params for a specific scene type without side effects."""
        param_dict = self._scene_params.get(scene_type, {})
        return DecodingParams(
            beam_size=param_dict.get("beam_size", 5),
            temperature=param_dict.get("temperature", 0.0),
            patience=param_dict.get("patience", 1.0),
            best_of=param_dict.get("best_of", 1),
            compression_ratio_threshold=param_dict.get(
                "compression_ratio_threshold", 2.4
            ),
            no_speech_threshold=param_dict.get("no_speech_threshold", 0.6),
            log_prob_threshold=param_dict.get("log_prob_threshold", -1.0),
            language=self._base_language,
            scene_type=scene_type,
        )
