"""
Adaptive decoding parameters — scene-driven STT configuration.

Encapsulates all tunable parameters that the Adaptive Decoder passes
to the STT engine.  Each acoustic scene maps to a different set of
parameters via ``config/defaults/scenes.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from speech.models.scene_type import SceneType


@dataclass
class DecodingParams:
    """Parameters controlling the STT engine's decoding behaviour.

    These are primarily faster-whisper / CTranslate2 parameters, but the
    interface is engine-agnostic — each engine maps applicable fields.

    Attributes:
        beam_size:             Number of beams for beam search decoding.
        temperature:           Sampling temperature (0.0 = greedy / deterministic).
        patience:              Beam search patience factor (1.0 = standard).
        best_of:               Number of candidates when sampling (temperature > 0).
        length_penalty:        Exponential length penalty (1.0 = no penalty).
        compression_ratio_threshold: Max compression ratio before rejecting.
        no_speech_threshold:   Probability threshold for no-speech detection.
        log_prob_threshold:    Minimum average log probability per token.
        initial_prompt:        Conditioning prompt for vocabulary biasing.
        language:              Language code (e.g. "en") or None for auto-detect.
        scene_type:            The acoustic scene that produced these params.
    """

    beam_size: int = 5
    temperature: float = 0.0
    patience: float = 1.0
    best_of: int = 1
    length_penalty: float = 1.0
    compression_ratio_threshold: float = 2.4
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    initial_prompt: Optional[str] = None
    language: Optional[str] = "en"
    scene_type: SceneType = SceneType.UNKNOWN

    # Hard bounds to prevent runaway values from misconfiguration
    _BEAM_MIN: int = 1
    _BEAM_MAX: int = 15
    _TEMP_MIN: float = 0.0
    _TEMP_MAX: float = 1.0
    _PATIENCE_MIN: float = 0.5
    _PATIENCE_MAX: float = 3.0

    def __post_init__(self):
        """Clamp all values to safe ranges."""
        self.beam_size = max(self._BEAM_MIN, min(self._BEAM_MAX, self.beam_size))
        self.temperature = max(self._TEMP_MIN, min(self._TEMP_MAX, self.temperature))
        self.patience = max(self._PATIENCE_MIN, min(self._PATIENCE_MAX, self.patience))
        self.best_of = max(1, min(10, self.best_of))

    @classmethod
    def for_quiet(cls) -> DecodingParams:
        """Optimised parameters for quiet environments."""
        return cls(
            beam_size=3,
            temperature=0.0,
            patience=1.0,
            best_of=1,
            no_speech_threshold=0.6,
            scene_type=SceneType.QUIET_ROOM,
        )

    @classmethod
    def for_moderate_noise(cls) -> DecodingParams:
        """Optimised parameters for moderate noise (office, classroom)."""
        return cls(
            beam_size=5,
            temperature=0.0,
            patience=1.2,
            best_of=3,
            no_speech_threshold=0.5,
            scene_type=SceneType.OFFICE,
        )

    @classmethod
    def for_noisy(cls) -> DecodingParams:
        """Optimised parameters for noisy environments (street, bus)."""
        return cls(
            beam_size=8,
            temperature=0.0,
            patience=1.5,
            best_of=5,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.5,
            scene_type=SceneType.STREET,
        )

    @classmethod
    def for_very_noisy(cls) -> DecodingParams:
        """Optimised parameters for very noisy environments."""
        return cls(
            beam_size=10,
            temperature=0.0,
            patience=2.0,
            best_of=5,
            no_speech_threshold=0.3,
            log_prob_threshold=-2.0,
            compression_ratio_threshold=3.0,
            scene_type=SceneType.VERY_NOISY,
        )

    def with_temperature(self, temperature: float) -> DecodingParams:
        """Return a copy with a different temperature (for redecoding)."""
        return DecodingParams(
            beam_size=self.beam_size,
            temperature=temperature,
            patience=self.patience,
            best_of=max(self.best_of, 3) if temperature > 0 else self.best_of,
            length_penalty=self.length_penalty,
            compression_ratio_threshold=self.compression_ratio_threshold,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            initial_prompt=self.initial_prompt,
            language=self.language,
            scene_type=self.scene_type,
        )

    def with_prompt(self, prompt: str) -> DecodingParams:
        """Return a copy with an updated initial_prompt (for vocabulary injection)."""
        return DecodingParams(
            beam_size=self.beam_size,
            temperature=self.temperature,
            patience=self.patience,
            best_of=self.best_of,
            length_penalty=self.length_penalty,
            compression_ratio_threshold=self.compression_ratio_threshold,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            initial_prompt=prompt,
            language=self.language,
            scene_type=self.scene_type,
        )
