"""
Unit tests for the Personal Adaptation module.

Tests cover:
- UserProfile data and persistence
- UserProfileManager switching and save/load
- CommandHistory frequency tracking and recency weighting
- CorrectionLearner learning and applying corrections
- AccentAdapter prompt generation
- HealthMonitor registration and checking
- Dashboard rendering
- EchoCanceller state management
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from speech.adaptation.accent_adapter import AccentAdapter
from speech.adaptation.command_history import CommandHistory, CommandRecord
from speech.adaptation.correction_learner import CorrectionLearner
from speech.adaptation.user_profile import UserProfile, UserProfileManager
from speech.monitoring.dashboard import Dashboard
from speech.monitoring.health_monitor import HealthMonitor, HealthState, HealthStatus
from speech.audio.echo_canceller import EchoCanceller
from speech.vocabulary.phonetic_dictionary import PhoneticDictionary


# ── UserProfile ─────────────────────────────────────────────

class TestUserProfile:
    def test_default_values(self):
        p = UserProfile()
        assert p.user_id == "default"
        assert p.preferred_speed == 1.0
        assert not p.enrolled

    def test_add_correction(self):
        p = UserProfile()
        p.add_correction("novice gate", "navigate")
        assert p.get_correction("novice gate") == "navigate"

    def test_to_dict_and_from_dict(self):
        p = UserProfile(user_id="test", accent_hint="indian english")
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.user_id == "test"
        assert p2.accent_hint == "indian english"


class TestUserProfileManager:
    def test_active_profile_default(self):
        mgr = UserProfileManager()
        profile = mgr.active_profile
        assert profile.user_id == "default"

    def test_set_active(self):
        mgr = UserProfileManager()
        p = mgr.set_active("alice")
        assert p.user_id == "alice"
        assert mgr.active_profile.user_id == "alice"

    def test_list_profiles(self):
        mgr = UserProfileManager()
        mgr.set_active("alice")
        mgr.set_active("bob")
        profiles = mgr.list_profiles()
        assert "alice" in profiles
        assert "bob" in profiles


# ── CommandHistory ──────────────────────────────────────────

class TestCommandHistory:
    def test_record_and_frequency(self):
        h = CommandHistory()
        h.record("navigate")
        h.record("navigate")
        h.record("stop")
        assert h.get_frequency_boost("navigate") > h.get_frequency_boost("stop")

    def test_top_commands(self):
        h = CommandHistory()
        for _ in range(5):
            h.record("navigate")
        for _ in range(3):
            h.record("stop")
        top = h.get_top_commands(2)
        assert top[0][0] == "navigate"
        assert top[0][1] == 5

    def test_success_rate(self):
        h = CommandHistory()
        h.record("test", success=True)
        h.record("test", success=True)
        h.record("test", success=False)
        rec = h.get_record("test")
        assert rec is not None
        assert 0.6 < rec.success_rate < 0.7

    def test_unknown_action_zero_boost(self):
        h = CommandHistory()
        assert h.get_frequency_boost("nonexistent") == 0.0

    def test_max_actions_eviction(self):
        h = CommandHistory(max_actions=3)
        h.record("a")
        h.record("b")
        h.record("c")
        h.record("d")  # Should evict least used
        assert h.size <= 3

    def test_recency_weight(self):
        r = CommandRecord(action="test", count=1, last_used=time.time())
        assert r.recency_weight > 0.9  # Recent → high weight


# ── CorrectionLearner ──────────────────────────────────────

class TestCorrectionLearner:
    def test_learn_and_apply(self):
        learner = CorrectionLearner()
        learner.learn("novice gate", "navigate")
        assert learner.apply_corrections("novice gate") == "navigate"

    def test_apply_unknown_no_change(self):
        learner = CorrectionLearner()
        assert learner.apply_corrections("hello world") == "hello world"

    def test_learn_updates_phonetic_dict(self):
        pd = PhoneticDictionary(entries={})
        learner = CorrectionLearner(phonetic_dict=pd)
        learner.learn("salmun", "Simon")
        assert pd.lookup("salmun") == "Simon"

    def test_learn_updates_profile(self):
        profile = UserProfile()
        learner = CorrectionLearner(profile=profile)
        learner.learn("wrong", "right")
        assert profile.get_correction("wrong") == "right"

    def test_correction_count(self):
        learner = CorrectionLearner()
        learner.learn("a", "b")
        learner.learn("c", "d")
        assert learner.correction_count >= 2


# ── AccentAdapter ──────────────────────────────────────────

class TestAccentAdapter:
    def test_get_accent_prompt(self):
        adapter = AccentAdapter(accent_hint="Indian English")
        prompt = adapter.get_accent_prompt()
        assert "Indian English" in prompt

    def test_no_accent_empty_prompt(self):
        adapter = AccentAdapter(accent_hint="")
        assert adapter.get_accent_prompt() == ""

    def test_unknown_accent_empty_prompt(self):
        adapter = AccentAdapter(accent_hint="Martian")
        assert adapter.get_accent_prompt() == ""

    def test_build_combined_prompt(self):
        adapter = AccentAdapter(accent_hint="Indian English")
        combined = adapter.build_combined_prompt("navigate stop help")
        assert "Indian English" in combined
        assert "navigate" in combined

    def test_add_custom_accent(self):
        adapter = AccentAdapter()
        adapter.add_accent("klingon", "Speaking Klingon English.")
        adapter.accent = "klingon"
        assert "Klingon" in adapter.get_accent_prompt()

    def test_list_accents(self):
        adapter = AccentAdapter()
        accents = adapter.list_accents()
        assert "indian english" in accents
        assert len(accents) >= 4


# ── HealthMonitor ──────────────────────────────────────────

class TestHealthMonitor:
    def test_register_and_check(self):
        monitor = HealthMonitor(check_interval_s=0)
        monitor.register("stt", lambda: HealthStatus(
            component="stt", state=HealthState.HEALTHY, message="OK"
        ))
        results = monitor.check()
        assert results["stt"].state == HealthState.HEALTHY

    def test_unhealthy_component(self):
        monitor = HealthMonitor(check_interval_s=0)
        monitor.register("vad", lambda: HealthStatus(
            component="vad", state=HealthState.UNHEALTHY, message="Model missing"
        ))
        monitor.check()
        assert not monitor.all_healthy
        assert "vad" in monitor.unhealthy_components

    def test_check_exception(self):
        monitor = HealthMonitor(check_interval_s=0)

        def failing_check():
            raise RuntimeError("Check exploded")

        monitor.register("bad", failing_check)
        results = monitor.check()
        assert results["bad"].state == HealthState.UNHEALTHY

    def test_get_summary(self):
        monitor = HealthMonitor(check_interval_s=0)
        monitor.register("stt", lambda: HealthStatus(
            component="stt", state=HealthState.HEALTHY, message="OK"
        ))
        monitor.check()
        summary = monitor.get_summary()
        assert "stt" in summary
        assert "HEALTHY" in summary["stt"]


# ── Dashboard ──────────────────────────────────────────────

class TestDashboard:
    def test_render_not_empty(self):
        dashboard = Dashboard()
        output = dashboard.render()
        assert "SIMON" in output
        assert "Pipelines" in output

    def test_pipeline_status(self):
        dashboard = Dashboard()
        dashboard.set_pipeline_status("listen", "running")
        output = dashboard.render()
        assert "running" in output

    def test_custom_panel(self):
        dashboard = Dashboard()
        dashboard.add_panel("STT Metrics", {"latency_ms": 45.2, "confidence": 0.92})
        output = dashboard.render()
        assert "STT Metrics" in output
        assert "45.200" in output


# ── EchoCanceller ──────────────────────────────────────────

class TestEchoCanceller:
    def test_no_echo_when_not_playing(self):
        ec = EchoCanceller()
        audio = np.random.randn(1600).astype(np.float32)
        result = ec.cancel(audio)
        np.testing.assert_array_equal(result, audio)

    def test_echo_suppression_when_playing(self):
        ec = EchoCanceller()
        ec.on_playback_start()

        # Feed a loud reference signal
        reference = np.ones(1600, dtype=np.float32) * 0.5
        ec.feed_reference(reference)

        # Microphone picks up a quiet echo
        mic = np.ones(1600, dtype=np.float32) * 0.1
        result = ec.cancel(mic)

        # Result should be suppressed (quieter than input)
        assert np.sqrt(np.mean(result ** 2)) <= np.sqrt(np.mean(mic ** 2))

    def test_reset(self):
        ec = EchoCanceller()
        ec.on_playback_start()
        ec.reset()
        assert not ec.is_playing
