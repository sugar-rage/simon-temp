"""
Unit tests for the AdaptiveDecoder.

Tests verify:
- Default params returned for unknown scene
- Correct params returned for each scene type
- Scene change triggers parameter update
- initial_prompt injection
- Params stay within DecodingParams hard bounds
"""

from __future__ import annotations

import pytest

from speech.models.decoding_params import DecodingParams
from speech.models.scene_type import SceneType
from speech.scene.scene_profile import SceneProfile
from speech.stt.adaptive_decoder import AdaptiveDecoder


class TestAdaptiveDecoder:
    """Tests for AdaptiveDecoder."""

    def test_default_params_for_unknown_scene(self):
        decoder = AdaptiveDecoder()
        params = decoder.get_params()

        assert isinstance(params, DecodingParams)
        assert params.language == "en"

    def test_quiet_room_uses_small_beam(self):
        decoder = AdaptiveDecoder()
        profile = SceneProfile(scene_type=SceneType.QUIET_ROOM, confidence=0.9)
        params = decoder.get_params(profile)

        assert params.beam_size == 3
        assert params.patience == 1.0
        assert params.best_of == 1

    def test_street_uses_large_beam(self):
        decoder = AdaptiveDecoder()
        profile = SceneProfile(scene_type=SceneType.STREET, confidence=0.8)
        params = decoder.get_params(profile)

        assert params.beam_size == 8
        assert params.patience == 1.5
        assert params.best_of == 5

    def test_very_noisy_uses_maximum_search(self):
        decoder = AdaptiveDecoder()
        profile = SceneProfile(scene_type=SceneType.VERY_NOISY, confidence=0.7)
        params = decoder.get_params(profile)

        assert params.beam_size == 10
        assert params.patience == 2.0
        assert params.best_of == 5
        assert params.no_speech_threshold == 0.3

    def test_scene_change_updates_params(self):
        decoder = AdaptiveDecoder()

        quiet = SceneProfile(scene_type=SceneType.QUIET_ROOM, confidence=0.9)
        params_quiet = decoder.get_params(quiet)

        street = SceneProfile(scene_type=SceneType.STREET, confidence=0.8)
        params_street = decoder.get_params(street)

        assert params_quiet.beam_size < params_street.beam_size
        assert params_quiet.best_of < params_street.best_of

    def test_initial_prompt_injected(self):
        decoder = AdaptiveDecoder()
        profile = SceneProfile(scene_type=SceneType.OFFICE, confidence=0.8)
        params = decoder.get_params(profile, initial_prompt="navigate read_text stop")

        assert params.initial_prompt == "navigate read_text stop"

    def test_current_params_property(self):
        decoder = AdaptiveDecoder()
        profile = SceneProfile(scene_type=SceneType.BUS_TRAIN, confidence=0.7)
        params = decoder.get_params(profile)

        assert decoder.current_params is params

    def test_get_params_for_scene_no_side_effects(self):
        decoder = AdaptiveDecoder()
        params = decoder.get_params_for_scene(SceneType.SHOPPING_MALL)

        assert params.scene_type == SceneType.SHOPPING_MALL
        # Should not update the internal last scene
        assert decoder._last_scene == SceneType.UNKNOWN

    def test_all_scene_types_produce_valid_params(self):
        """Every known SceneType should produce valid DecodingParams."""
        decoder = AdaptiveDecoder()

        for scene in SceneType:
            params = decoder.get_params_for_scene(scene)
            assert isinstance(params, DecodingParams)
            assert 1 <= params.beam_size <= 15
            assert 0.0 <= params.temperature <= 1.0
            assert 0.5 <= params.patience <= 3.0

    def test_custom_scene_params(self):
        """Custom scene-to-params mapping overrides defaults."""
        custom = {
            SceneType.QUIET_ROOM: dict(beam_size=1, patience=0.5, best_of=1),
        }
        decoder = AdaptiveDecoder(scene_params=custom)
        profile = SceneProfile(scene_type=SceneType.QUIET_ROOM, confidence=0.9)
        params = decoder.get_params(profile)

        # beam_size=1 gets clamped to min=1 by DecodingParams.__post_init__
        assert params.beam_size == 1
        assert params.patience == 0.5

    def test_language_preserved(self):
        decoder = AdaptiveDecoder(base_language="hi")
        profile = SceneProfile(scene_type=SceneType.STREET, confidence=0.8)
        params = decoder.get_params(profile)

        assert params.language == "hi"

    def test_each_scene_maps_to_correct_no_speech_threshold(self):
        decoder = AdaptiveDecoder()

        quiet_params = decoder.get_params_for_scene(SceneType.QUIET_ROOM)
        noisy_params = decoder.get_params_for_scene(SceneType.VERY_NOISY)

        # Noisy scenes should have lower no_speech_threshold (more lenient)
        assert noisy_params.no_speech_threshold < quiet_params.no_speech_threshold
