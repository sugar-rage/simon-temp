"""
Unit tests for the SceneClassifier (heuristic acoustic scene classifier).

Tests verify:
- Silent audio classifies as QUIET_ROOM
- Loud broadband noise classifies as a noisy scene
- EMA smoothing prevents rapid flipping
- Majority voting stabilises output
- Throttling prevents over-classification
- Reset clears all state
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from speech.models.audio_frame import AudioFrame
from speech.models.scene_type import SceneType
from speech.scene.scene_classifier import SceneClassifier
from speech.scene.scene_profile import SceneProfile


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #

def _make_frame(
    amplitude: float = 0.0,
    frequency: float = 0.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    add_noise: bool = False,
    noise_amplitude: float = 0.01,
) -> AudioFrame:
    """Create a test audio frame with optional sine wave and noise."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)

    if frequency > 0 and amplitude > 0:
        data = amplitude * np.sin(2 * np.pi * frequency * t)
    else:
        data = np.zeros(n, dtype=np.float32)

    if add_noise:
        rng = np.random.default_rng(42)
        data += rng.uniform(-noise_amplitude, noise_amplitude, n)

    return AudioFrame(
        data=data.astype(np.float32),
        sample_rate=sample_rate,
        dtype="float32",
    )


# ---------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------- #

class TestSceneClassifier:
    """Tests for SceneClassifier."""

    def test_init_defaults(self):
        clf = SceneClassifier()
        assert clf.name == "heuristic-scene-classifier"

    def test_silent_audio_classified_as_quiet(self):
        clf = SceneClassifier(classification_interval_s=0.0)
        frame = _make_frame(amplitude=0.0, duration_s=1.0)
        profile = clf.classify(frame)

        assert isinstance(profile, SceneProfile)
        assert profile.scene_type == SceneType.QUIET_ROOM
        assert profile.confidence > 0.5

    def test_loud_broadband_noise_classified_as_noisy(self):
        clf = SceneClassifier(classification_interval_s=0.0)
        # High amplitude + high ZCR = noisy
        frame = _make_frame(
            amplitude=0.0,
            add_noise=True,
            noise_amplitude=0.3,
            duration_s=1.0,
        )
        profile = clf.classify(frame)

        assert profile.scene_type.is_noisy or profile.scene_type == SceneType.VERY_NOISY

    def test_moderate_sine_classified_as_indoor(self):
        clf = SceneClassifier(classification_interval_s=0.0)
        # Moderate amplitude, low frequency sine
        frame = _make_frame(amplitude=0.04, frequency=300, duration_s=1.0)
        profile = clf.classify(frame)

        # Should be some indoor scene, not VERY_NOISY
        assert profile.scene_type != SceneType.VERY_NOISY

    def test_classification_throttling(self):
        """Second call within interval returns cached profile."""
        clf = SceneClassifier(classification_interval_s=10.0)

        frame1 = _make_frame(amplitude=0.0, duration_s=1.0)
        profile1 = clf.classify(frame1)

        # Second call immediately — should return same cached profile
        loud_frame = _make_frame(amplitude=0.3, add_noise=True, noise_amplitude=0.3)
        profile2 = clf.classify(loud_frame)

        assert profile2.timestamp == profile1.timestamp  # Same cached result

    def test_reset_clears_state(self):
        clf = SceneClassifier(classification_interval_s=0.0)

        # Classify something
        frame = _make_frame(amplitude=0.04, frequency=300, duration_s=1.0)
        clf.classify(frame)

        # Reset
        clf.reset()

        # Internal state should be cleared
        assert clf._initialized is False
        assert len(clf._history) == 0

    def test_history_voting_stabilises_output(self):
        """Majority voting prevents a single outlier from flipping the scene."""
        clf = SceneClassifier(
            classification_interval_s=0.0,
            history_size=5,
        )

        # Feed 4 quiet frames
        quiet = _make_frame(amplitude=0.001, duration_s=1.0)
        for _ in range(4):
            clf.classify(quiet)

        # Feed 1 noisy frame
        noisy = _make_frame(amplitude=0.0, add_noise=True, noise_amplitude=0.3)
        profile = clf.classify(noisy)

        # Majority should still be QUIET_ROOM (4 vs 1)
        assert profile.scene_type == SceneType.QUIET_ROOM

    def test_profile_has_valid_snr(self):
        clf = SceneClassifier(classification_interval_s=0.0)
        frame = _make_frame(amplitude=0.04, frequency=300, duration_s=1.0)
        profile = clf.classify(frame)

        assert -10.0 <= profile.estimated_snr_db <= 60.0

    def test_profile_has_valid_reverb(self):
        clf = SceneClassifier(classification_interval_s=0.0)
        frame = _make_frame(amplitude=0.04, frequency=300, duration_s=1.0)
        profile = clf.classify(frame)

        assert 0.0 <= profile.reverb_estimate <= 1.0

    def test_scene_profile_is_noisy_property(self):
        profile = SceneProfile(scene_type=SceneType.QUIET_ROOM, estimated_snr_db=30.0)
        assert not profile.is_noisy

        profile = SceneProfile(scene_type=SceneType.STREET, estimated_snr_db=10.0)
        assert profile.is_noisy

    def test_scene_profile_is_reliable_property(self):
        profile = SceneProfile(confidence=0.3)
        assert not profile.is_reliable

        profile = SceneProfile(confidence=0.7)
        assert profile.is_reliable

    def test_ema_smoothing_applied(self):
        """EMA should smooth feature values over time."""
        clf = SceneClassifier(
            classification_interval_s=0.0,
            ema_alpha=0.5,
        )

        # First frame initialises EMA
        frame1 = _make_frame(amplitude=0.04, frequency=300, duration_s=1.0)
        clf.classify(frame1)
        rms1 = clf._ema_rms

        # Second frame with different amplitude
        frame2 = _make_frame(amplitude=0.08, frequency=300, duration_s=1.0)
        clf.classify(frame2)
        rms2 = clf._ema_rms

        # EMA should be between the two raw values
        assert rms2 != rms1  # Not the same — EMA moved
