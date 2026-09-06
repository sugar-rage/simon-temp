"""
Unit tests for the SafetyValidator.

Tests verify:
- Safe actions pass immediately
- Cautious actions pass with sufficient confidence
- Cautious actions rejected with low confidence
- Dangerous arguments trigger confirmation
- Blocked patterns block execution
- Unknown actions apply default confidence
- ConfirmationManager state transitions
"""

from __future__ import annotations

import time

import pytest

from speech.safety.confirmation_manager import ConfirmationManager, ConfirmationState
from speech.safety.safety_rules import SafetyLevel, SafetyRule
from speech.safety.safety_validator import SafetyValidator, ValidationDecision


class TestSafetyValidator:
    """Tests for SafetyValidator."""

    def test_safe_action_passes(self):
        validator = SafetyValidator()
        decision = validator.validate("read_text", confidence=0.60)

        assert decision.allowed
        assert decision.safety_level == SafetyLevel.SAFE

    def test_safe_action_low_confidence_still_passes(self):
        """Safe actions have low min_confidence thresholds."""
        validator = SafetyValidator()
        decision = validator.validate("help", confidence=0.42)

        assert decision.allowed

    def test_cautious_action_passes_with_good_confidence(self):
        validator = SafetyValidator()
        decision = validator.validate("navigate", confidence=0.85, args={"destination": "library"})

        assert decision.allowed
        assert decision.safety_level == SafetyLevel.CAUTIOUS

    def test_cautious_action_rejected_low_confidence(self):
        validator = SafetyValidator()
        decision = validator.validate("navigate", confidence=0.50, args={"destination": "library"})

        assert not decision.allowed
        assert "Confidence" in decision.reason

    def test_dangerous_args_trigger_confirmation(self):
        validator = SafetyValidator()
        decision = validator.validate(
            "navigate", confidence=0.85, args={"destination": "highway 101"}
        )

        assert not decision.allowed
        assert decision.needs_confirm
        assert decision.safety_level == SafetyLevel.DANGEROUS

    def test_emergency_call_always_requires_confirmation(self):
        validator = SafetyValidator()
        decision = validator.validate("emergency_call", confidence=0.90)

        assert not decision.allowed
        assert decision.needs_confirm
        assert decision.safety_level == SafetyLevel.DANGEROUS

    def test_unknown_action_default_confidence(self):
        validator = SafetyValidator(default_min_confidence=0.60)
        decision = validator.validate("unknown_action", confidence=0.65)

        assert decision.allowed

    def test_unknown_action_low_confidence_rejected(self):
        validator = SafetyValidator(default_min_confidence=0.60)
        decision = validator.validate("unknown_action", confidence=0.40)

        assert not decision.allowed

    def test_add_rule_at_runtime(self):
        validator = SafetyValidator()
        rule = SafetyRule(
            action="custom_action",
            level=SafetyLevel.DANGEROUS,
            min_confidence=0.90,
            confirm_message="Confirm custom action?",
        )
        validator.add_rule(rule)

        decision = validator.validate("custom_action", confidence=0.95)
        assert decision.needs_confirm

    def test_get_rule(self):
        validator = SafetyValidator()
        rule = validator.get_rule("navigate")
        assert rule is not None
        assert rule.action == "navigate"

    def test_blocked_pattern_blocks_action(self):
        rule = SafetyRule(
            action="test_action",
            level=SafetyLevel.CAUTIOUS,
            min_confidence=0.50,
            blocked_patterns=[r"\bdangerous_word\b"],
        )
        validator = SafetyValidator(rules={"test_action": rule})

        decision = validator.validate(
            "test_action", confidence=0.90,
            args={"target": "dangerous_word"}
        )

        assert not decision.allowed
        assert decision.blocked


class TestConfirmationManager:
    """Tests for ConfirmationManager."""

    def test_initial_state_is_idle(self):
        mgr = ConfirmationManager()
        assert mgr.state == ConfirmationState.IDLE

    def test_request_confirmation_sets_waiting(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Navigate to highway?")
        assert mgr.state == ConfirmationState.WAITING
        assert mgr.pending_action == "navigate"

    def test_yes_response_confirms(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        result = mgr.process_response("yes")
        assert result == ConfirmationState.CONFIRMED

    def test_no_response_denies(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        result = mgr.process_response("no")
        assert result == ConfirmationState.DENIED

    def test_cancel_response(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        result = mgr.process_response("cancel")
        assert result == ConfirmationState.DENIED

    def test_ambiguous_response_stays_waiting(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        result = mgr.process_response("maybe later")
        assert result == ConfirmationState.WAITING

    def test_timeout(self):
        mgr = ConfirmationManager(timeout_s=0.01)
        mgr.request_confirmation("navigate", "Confirm?")
        time.sleep(0.02)
        assert mgr.state == ConfirmationState.TIMED_OUT

    def test_cancel_method(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        mgr.cancel()
        assert mgr.state == ConfirmationState.CANCELLED

    def test_reset(self):
        mgr = ConfirmationManager()
        mgr.request_confirmation("navigate", "Confirm?")
        mgr.reset()
        assert mgr.state == ConfirmationState.IDLE
        assert mgr.pending_action is None

    def test_process_response_when_not_waiting(self):
        mgr = ConfirmationManager()
        result = mgr.process_response("yes")
        assert result == ConfirmationState.IDLE

    def test_speak_fn_called_on_request(self):
        spoken = []
        mgr = ConfirmationManager(speak_fn=lambda t: spoken.append(t))
        mgr.request_confirmation("nav", "Navigate?")
        assert "Navigate?" in spoken

    def test_is_waiting_property(self):
        mgr = ConfirmationManager()
        assert not mgr.is_waiting
        mgr.request_confirmation("nav", "Confirm?")
        assert mgr.is_waiting
