"""
Confidence gate — enforces per-action confidence thresholds.

The confidence gate sits after the STT engine and before intent parsing.
It checks whether the transcript confidence meets the required threshold
for the detected action.  Safety-critical actions (navigation, stop)
require higher confidence than informational queries (status, help).

Design decision: Per-action thresholds rather than a single global threshold.
This allows:
- "stop" to be accepted at 0.75 (common word, easy to recognise)
- "navigate to" to require 0.85 (safety-critical, misrecognition is dangerous)
- "what time is it" to accept 0.60 (low-risk query)
"""

from __future__ import annotations

from typing import Optional

from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.stt.transcript import Transcript

logger = get_logger("stt.confidence_gate")
metrics = get_collector()


# Default per-action thresholds (can be overridden via config)
_DEFAULT_THRESHOLDS = {
    # Safety-critical
    "navigate": 0.85,
    "cancel_nav": 0.80,
    "save_face": 0.70,
    "stop": 0.75,

    # Perception
    "read_text": 0.65,
    "identify_face": 0.65,
    "describe": 0.60,

    # Informational / Conversational
    "status": 0.55,
    "help": 0.50,
    "repeat": 0.55,
    "battery": 0.55,
    "time": 0.55,
    "volume_up": 0.60,
    "volume_down": 0.60,
    "toggle_indoor": 0.65,
    "yes": 0.50,
    "no": 0.50,
    "no_name": 0.50,
    "unmatched_text": 0.50,
    "name": 0.40,
    "confirmation": 0.30,
}


class ConfidenceGateResult:
    """Result of a confidence gate check.

    Attributes:
        passed:     True if the transcript meets the confidence threshold.
        action:     The detected action (if known at this stage).
        threshold:  The threshold that was applied.
        confidence: The transcript's confidence score.
        reason:     Human-readable reason if the gate blocked the transcript.
    """

    __slots__ = ("passed", "action", "threshold", "confidence", "reason")

    def __init__(
        self,
        passed: bool,
        action: str = "",
        threshold: float = 0.0,
        confidence: float = 0.0,
        reason: str = "",
    ):
        self.passed = passed
        self.action = action
        self.threshold = threshold
        self.confidence = confidence
        self.reason = reason


class ConfidenceGate:
    """Enforces per-action confidence thresholds on STT transcripts.

    Args:
        thresholds:             Dict mapping action names to confidence thresholds.
        default_threshold:      Fallback threshold for unknown actions.
        safety_threshold:       Minimum threshold for safety-critical actions.
        expecting_name:         Whether the system is expecting a spoken name.
        expecting_confirmation: Whether the system is expecting a confirmation (yes/no).
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        default_threshold: float = 0.60,
        safety_threshold: float = 0.85,
        expecting_name: bool = False,
        expecting_confirmation: bool = False,
    ):
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self._thresholds.update(thresholds)
        self._default = default_threshold
        self._safety = safety_threshold
        self._expecting_name = expecting_name
        self._expecting_confirmation = expecting_confirmation

    @property
    def expecting_name(self) -> bool:
        """Whether the confidence gate is currently in name-prompting context."""
        return self._expecting_name

    @expecting_name.setter
    def expecting_name(self, value: bool) -> None:
        self._expecting_name = bool(value)

    def set_expecting_name(self, expecting: bool) -> None:
        """Set whether the system is expecting a spoken name during face registration."""
        logger.info("[NAME_TRACE] ConfidenceGate set_expecting_name: %s -> %s", self._expecting_name, expecting)
        self._expecting_name = bool(expecting)

    @property
    def expecting_confirmation(self) -> bool:
        """Whether the confidence gate is currently in confirmation-prompting context."""
        return self._expecting_confirmation

    @expecting_confirmation.setter
    def expecting_confirmation(self, value: bool) -> None:
        self._expecting_confirmation = bool(value)

    def set_expecting_confirmation(self, expecting: bool) -> None:
        """Set whether the system is expecting a confirmation response during face registration."""
        logger.info("[CONFIRM_TRACE] ConfidenceGate set_expecting_confirmation: %s -> %s", self._expecting_confirmation, expecting)
        self._expecting_confirmation = bool(expecting)

    def get_threshold(self, action: str) -> float:
        """Get the confidence threshold for a given action."""
        if self._expecting_confirmation and action in ("yes", "no", "confirmation", "unmatched_text"):
            t = self._thresholds.get("confirmation", 0.30)
            logger.info("[CONFIRM_TRACE] ConfidenceGate resolving threshold for %r with expecting_confirmation=True -> %.2f", action, t)
            return t
        if self._expecting_name and action in ("unmatched_text", "name"):
            t = self._thresholds.get("name", 0.40)
            logger.info("[NAME_TRACE] ConfidenceGate resolving threshold for %r with expecting_name=True -> %.2f", action, t)
            return t
        t = self._thresholds.get(action, self._default)
        if self._expecting_confirmation:
            logger.info("[CONFIRM_TRACE] ConfidenceGate resolving threshold for %r with expecting_confirmation=%s -> %.2f", action, self._expecting_confirmation, t)
        elif self._expecting_name:
            logger.info("[NAME_TRACE] ConfidenceGate resolving threshold for %r with expecting_name=%s -> %.2f", action, self._expecting_name, t)
        return t

    def set_threshold(self, action: str, threshold: float) -> None:
        """Update the threshold for a specific action."""
        self._thresholds[action] = max(0.1, min(1.0, threshold))

    def check(
        self,
        transcript: Transcript,
        action: Optional[str] = None,
    ) -> ConfidenceGateResult:
        """Check whether a transcript passes the confidence gate.

        Args:
            transcript: The STT transcript to evaluate.
            action:     The detected action name.  If None, uses default threshold.

        Returns:
            ``ConfidenceGateResult`` indicating pass/fail and reason.
        """
        if transcript.is_empty:
            return ConfidenceGateResult(
                passed=False,
                action=action or "",
                threshold=self._default,
                confidence=0.0,
                reason="Empty transcript",
            )

        threshold = self.get_threshold(action) if action else self._default
        confidence = transcript.confidence
        passed = confidence >= threshold

        logger.info(
            "[NAME_TRACE] ConfidenceGate evaluation: action=%r, text=%r, conf=%.2f, threshold=%.2f, passed=%s, expecting_name=%s",
            action, transcript.text, confidence, threshold, passed, self._expecting_name,
        )
        if self._expecting_confirmation:
            logger.info(
                "[CONFIRM_TRACE] ConfidenceGate evaluation: action=%r, text=%r, conf=%.2f, threshold=%.2f, passed=%s, expecting_confirmation=%s",
                action, transcript.text, confidence, threshold, passed, self._expecting_confirmation,
            )

        if not passed:
            logger.info(
                f"Confidence gate blocked",
                extra={
                    "action": action or "unknown",
                    "confidence": f"{confidence:.2f}",
                    "threshold": f"{threshold:.2f}",
                },
            )
            metrics.counter("stt.confidence_gate_blocks")

        result = ConfidenceGateResult(
            passed=passed,
            action=action or "",
            threshold=threshold,
            confidence=confidence,
            reason="" if passed else (
                f"Confidence {confidence:.2f} below threshold {threshold:.2f} "
                f"for action '{action or 'unknown'}'"
            ),
        )

        return result

    def check_blanket(self, transcript: Transcript) -> ConfidenceGateResult:
        """Check against the default threshold (action not yet known).

        Used as a quick pre-filter before intent parsing.  The full
        per-action check is done after the intent parser identifies
        the action.
        """
        return self.check(transcript, action=None)
