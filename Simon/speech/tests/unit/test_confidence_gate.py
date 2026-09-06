"""
Unit tests for the confidence gate.

Tests per-action thresholds, blanket pre-filtering, edge cases,
and configurable threshold updates.
"""

from __future__ import annotations

import pytest

from speech.stt.confidence_gate import ConfidenceGate, ConfidenceGateResult
from speech.stt.transcript import Transcript


# ---------------------------------------------------------------------- #
#  Fixtures
# ---------------------------------------------------------------------- #

def _make_transcript(text: str, confidence: float) -> Transcript:
    return Transcript(
        text=text,
        raw_text=text,
        confidence=confidence,
        engine_name="test",
    )


# ---------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------- #

class TestConfidenceGate:
    """Tests for the ConfidenceGate module."""

    def test_high_confidence_passes(self):
        gate = ConfidenceGate()
        t = _make_transcript("navigate to library", 0.92)
        result = gate.check(t, action="navigate")
        assert result.passed is True

    def test_low_confidence_blocked(self):
        gate = ConfidenceGate()
        t = _make_transcript("navigate to library", 0.50)
        result = gate.check(t, action="navigate")
        assert result.passed is False
        assert "below threshold" in result.reason

    def test_navigate_requires_high_threshold(self):
        gate = ConfidenceGate()
        threshold = gate.get_threshold("navigate")
        assert threshold == 0.85

    def test_help_requires_low_threshold(self):
        gate = ConfidenceGate()
        threshold = gate.get_threshold("help")
        assert threshold == 0.50

    def test_unknown_action_uses_default(self):
        gate = ConfidenceGate(default_threshold=0.60)
        threshold = gate.get_threshold("unknown_action")
        assert threshold == 0.60

    def test_empty_transcript_blocked(self):
        gate = ConfidenceGate()
        t = Transcript(text="", confidence=0.99)
        result = gate.check(t, action="status")
        assert result.passed is False
        assert "Empty" in result.reason

    def test_blanket_check_uses_default(self):
        gate = ConfidenceGate(default_threshold=0.60)
        t = _make_transcript("some text", 0.65)
        result = gate.check_blanket(t)
        assert result.passed is True

    def test_blanket_check_blocks_below_default(self):
        gate = ConfidenceGate(default_threshold=0.60)
        t = _make_transcript("some text", 0.40)
        result = gate.check_blanket(t)
        assert result.passed is False

    def test_custom_thresholds_override(self):
        gate = ConfidenceGate(thresholds={"navigate": 0.95})
        assert gate.get_threshold("navigate") == 0.95

    def test_set_threshold(self):
        gate = ConfidenceGate()
        gate.set_threshold("navigate", 0.90)
        assert gate.get_threshold("navigate") == 0.90

    def test_set_threshold_bounded(self):
        gate = ConfidenceGate()
        gate.set_threshold("navigate", 1.5)  # Above max
        assert gate.get_threshold("navigate") == 1.0
        gate.set_threshold("navigate", -0.5)  # Below min
        assert gate.get_threshold("navigate") == 0.1

    def test_result_attributes(self):
        gate = ConfidenceGate()
        t = _make_transcript("stop", 0.80)
        result = gate.check(t, action="stop")
        assert isinstance(result, ConfidenceGateResult)
        assert result.action == "stop"
        assert result.confidence == 0.80
        assert result.threshold == gate.get_threshold("stop")

    def test_stop_action_threshold(self):
        """Stop should require 0.75 (lower than navigate, higher than help)."""
        gate = ConfidenceGate()
        t = gate.get_threshold("stop")
        assert 0.50 < t < 0.85

    @pytest.mark.parametrize("action,expected_threshold", [
        ("navigate", 0.85),
        ("cancel_nav", 0.80),
        ("stop", 0.75),
        ("save_face", 0.70),
        ("read_text", 0.65),
        ("status", 0.55),
        ("help", 0.50),
    ])
    def test_all_default_thresholds(self, action, expected_threshold):
        gate = ConfidenceGate()
        assert gate.get_threshold(action) == expected_threshold
