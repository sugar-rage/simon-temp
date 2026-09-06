"""
Safety Validator — pre-execution safety gate for commands.

For every command the ListenPipeline produces, the SafetyValidator
determines whether the command can be executed immediately, needs
voice confirmation, or must be blocked.

Safety-critical decisions for a visually impaired user:
- "Navigate to highway" → requires confirmation (DANGEROUS destination)
- "Call 911" → requires confirmation (emergency service)
- "Read text" → passes immediately (SAFE action)
- Low-confidence "stop" → passes (SAFE, low threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from speech.safety.confirmation_manager import ConfirmationManager, ConfirmationState
from speech.safety.safety_rules import (
    DEFAULT_SAFETY_RULES,
    SafetyLevel,
    SafetyRule,
)

logger = logging.getLogger(__name__)


class ValidationDecision:
    """Result of a safety validation check.

    Attributes:
        allowed:        True if the command may proceed.
        needs_confirm:  True if confirmation was requested.
        blocked:        True if the command was blocked.
        reason:         Human-readable explanation.
        safety_level:   The determined safety level.
        rule:           The SafetyRule that was applied.
    """

    __slots__ = ("allowed", "needs_confirm", "blocked", "reason", "safety_level", "rule")

    def __init__(
        self,
        allowed: bool,
        needs_confirm: bool = False,
        blocked: bool = False,
        reason: str = "",
        safety_level: SafetyLevel = SafetyLevel.SAFE,
        rule: Optional[SafetyRule] = None,
    ):
        self.allowed = allowed
        self.needs_confirm = needs_confirm
        self.blocked = blocked
        self.reason = reason
        self.safety_level = safety_level
        self.rule = rule


class SafetyValidator:
    """Pre-execution safety gate.

    Args:
        rules:           Custom rules dict. None → use DEFAULT_SAFETY_RULES.
        confirmation_mgr: ConfirmationManager instance for voice dialogs.
        default_min_confidence: Fallback min confidence for unknown actions.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, SafetyRule]] = None,
        confirmation_mgr: Optional[ConfirmationManager] = None,
        default_min_confidence: float = 0.50,
    ):
        self._rules = dict(rules) if rules else dict(DEFAULT_SAFETY_RULES)
        self._confirmation = confirmation_mgr or ConfirmationManager()
        self._default_min_confidence = default_min_confidence

    @property
    def confirmation_manager(self) -> ConfirmationManager:
        """Access the underlying ConfirmationManager."""
        return self._confirmation

    def validate(
        self,
        action: str,
        confidence: float,
        args: Optional[Dict] = None,
    ) -> ValidationDecision:
        """Validate a command before execution.

        Args:
            action:     Action name (e.g. "navigate").
            confidence: STT confidence for this command.
            args:       Parsed command arguments.

        Returns:
            A ValidationDecision.
        """
        rule = self._rules.get(action)

        if rule is None:
            # Unknown action — apply default confidence check only
            if confidence < self._default_min_confidence:
                return ValidationDecision(
                    allowed=False,
                    reason=f"Confidence {confidence:.2f} below threshold "
                           f"{self._default_min_confidence:.2f} for unknown action '{action}'",
                    safety_level=SafetyLevel.CAUTIOUS,
                )
            return ValidationDecision(allowed=True, reason="No safety rule; default allow")

        # 1. Check confidence threshold
        if confidence < rule.min_confidence:
            return ValidationDecision(
                allowed=False,
                reason=f"Confidence {confidence:.2f} below minimum "
                       f"{rule.min_confidence:.2f} for '{action}'",
                safety_level=rule.level,
                rule=rule,
            )

        # 2. Determine effective safety level (may be elevated by args)
        effective_level = self._determine_level(rule, args)

        # 3. Handle by level
        if effective_level == SafetyLevel.BLOCKED:
            return ValidationDecision(
                allowed=False,
                blocked=True,
                reason=f"Action '{action}' is blocked by safety rule",
                safety_level=SafetyLevel.BLOCKED,
                rule=rule,
            )

        if effective_level == SafetyLevel.DANGEROUS:
            # Format confirmation message with args
            message = self._format_confirm_message(rule, args)
            self._confirmation.request_confirmation(action, message)

            return ValidationDecision(
                allowed=False,
                needs_confirm=True,
                reason=f"Action '{action}' requires confirmation",
                safety_level=SafetyLevel.DANGEROUS,
                rule=rule,
            )

        if effective_level == SafetyLevel.CAUTIOUS:
            # Cautious actions pass if confidence is sufficient (already checked)
            return ValidationDecision(
                allowed=True,
                reason=f"Cautious action '{action}' passed with confidence {confidence:.2f}",
                safety_level=SafetyLevel.CAUTIOUS,
                rule=rule,
            )

        # SAFE — always allow
        return ValidationDecision(
            allowed=True,
            reason=f"Safe action '{action}'",
            safety_level=SafetyLevel.SAFE,
            rule=rule,
        )

    def add_rule(self, rule: SafetyRule) -> None:
        """Add or update a safety rule at runtime."""
        self._rules[rule.action] = rule
        logger.info(f"Safety rule added/updated: {rule.action} → {rule.level.name}")

    def get_rule(self, action: str) -> Optional[SafetyRule]:
        """Get the safety rule for an action."""
        return self._rules.get(action)

    def _determine_level(
        self,
        rule: SafetyRule,
        args: Optional[Dict],
    ) -> SafetyLevel:
        """Check if arguments elevate the safety level."""
        if args is None:
            return rule.level

        args_str = " ".join(str(v) for v in args.values())

        # Check blocked patterns first
        for pattern in rule.get_blocked_patterns():
            if pattern.search(args_str):
                logger.warning(
                    f"Action '{rule.action}' BLOCKED by pattern '{pattern.pattern}' "
                    f"in args: {args_str}"
                )
                return SafetyLevel.BLOCKED

        # Check dangerous patterns
        for pattern in rule.get_dangerous_patterns():
            if pattern.search(args_str):
                logger.info(
                    f"Action '{rule.action}' elevated to DANGEROUS by pattern "
                    f"'{pattern.pattern}' in args: {args_str}"
                )
                return SafetyLevel.DANGEROUS

        return rule.level

    @staticmethod
    def _format_confirm_message(
        rule: SafetyRule,
        args: Optional[Dict],
    ) -> str:
        """Format the confirmation message with argument values."""
        message = rule.confirm_message
        if args:
            try:
                message = message.format(**args)
            except (KeyError, IndexError):
                pass  # Template didn't match args — use as-is
        return message
