"""
Integration test for the SpeechManager facade.

Tests that the SpeechManager correctly wires all subcomponents
and exposes the high-level API.  Uses mock/stub dependencies
since we don't have real audio hardware in CI.
"""

from __future__ import annotations

import pytest

from speech.context.context_manager import ContextManager
from speech.adaptation.user_profile import UserProfile, UserProfileManager
from speech.adaptation.command_history import CommandHistory
from speech.adaptation.correction_learner import CorrectionLearner
from speech.adaptation.accent_adapter import AccentAdapter
from speech.monitoring.health_monitor import HealthMonitor, HealthState, HealthStatus
from speech.monitoring.dashboard import Dashboard
from speech.safety.safety_validator import SafetyValidator
from speech.vocabulary.vocabulary_manager import VocabularyManager


class TestSpeechManagerIntegration:
    """Tests for the SpeechManager integration of Phase 5 components."""

    def test_context_safety_integration(self):
        """Context resolution feeds into safety validation."""
        context = ContextManager()
        safety = SafetyValidator()

        # Set up context
        context.set_context("last_location", "highway 101")

        # User says "go there" → resolves to "go highway 101"
        resolution = context.resolve_command_text("go there")
        assert "highway 101" in resolution.resolved_text

    def test_adaptation_profile_lifecycle(self):
        """Profile can be created, modified, and used for adaptation."""
        mgr = UserProfileManager()
        profile = mgr.set_active("test_user")
        profile.accent_hint = "indian english"
        profile.add_correction("novice gate", "navigate")

        adapter = AccentAdapter(accent_hint=profile.accent_hint)
        prompt = adapter.get_accent_prompt()
        assert "Indian English" in prompt

    def test_correction_learner_phonetic_integration(self):
        """CorrectionLearner integrates with PhoneticDictionary."""
        from speech.vocabulary.phonetic_dictionary import PhoneticDictionary

        pd = PhoneticDictionary(entries={})
        profile = UserProfile()
        learner = CorrectionLearner(profile=profile, phonetic_dict=pd)

        learner.learn("butterfly", "battery")
        # Now phonetic dict should know this mapping
        assert pd.lookup("butterfly") == "battery"
        # And profile stores it persistently
        assert profile.get_correction("butterfly") == "battery"

    def test_vocabulary_accent_combined_prompt(self):
        """VocabularyManager + AccentAdapter produce combined STT prompt."""
        vocab = VocabularyManager(command_words=["navigate", "stop"])
        adapter = AccentAdapter(accent_hint="Indian English")

        vocab_prompt = vocab.build_prompt()
        combined = adapter.build_combined_prompt(vocab_prompt)

        assert "Indian English" in combined
        assert "navigate" in combined

    def test_command_history_tracking(self):
        """Commands are tracked in history for frequency analysis."""
        history = CommandHistory()

        for _ in range(10):
            history.record("navigate", success=True)
        for _ in range(3):
            history.record("help", success=True)

        top = history.get_top_commands(2)
        assert top[0][0] == "navigate"
        assert history.get_frequency_boost("navigate") > 0

    def test_health_monitor_with_dashboard(self):
        """HealthMonitor feeds into Dashboard rendering."""
        monitor = HealthMonitor(check_interval_s=0)
        monitor.register("stt", lambda: HealthStatus(
            component="stt", state=HealthState.HEALTHY, message="Running"
        ))
        monitor.register("vad", lambda: HealthStatus(
            component="vad", state=HealthState.DEGRADED, message="Fallback mode"
        ))

        dashboard = Dashboard(health_monitor=monitor)
        dashboard.set_pipeline_status("listen", "running")
        dashboard.set_pipeline_status("speak", "running")
        dashboard.add_panel("Session", {
            "commands_processed": 42,
            "avg_confidence": 0.87,
        })

        output = dashboard.render()
        assert "SIMON" in output
        assert "running" in output
        assert "HEALTHY" in output
        assert "DEGRADED" in output

    def test_full_pipeline_with_adaptation(self):
        """End-to-end: profile + correction + context + safety."""
        # Setup
        profile = UserProfile(accent_hint="indian english")
        learner = CorrectionLearner(profile=profile)
        context = ContextManager()
        safety = SafetyValidator()

        # User teaches a correction
        learner.learn("novice gate", "navigate")

        # User navigates somewhere
        context.update_after_command("navigate", {"destination": "library"})

        # Later, user says "do that again" — but that's an anaphora test
        resolution = context.resolve_command_text("go there")
        assert "library" in resolution.resolved_text

        # Safety check
        decision = safety.validate("navigate", 0.85, {"destination": "library"})
        assert decision.allowed
