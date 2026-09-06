"""
Safety Rules — per-action safety rule definitions.

Each action in the SIMON command map can have an associated safety rule
that specifies:
- Whether the action requires voice confirmation before execution
- Minimum confidence threshold (overrides the global default)
- Dangerous argument patterns that trigger elevated confirmation
- Whether the action is blocked entirely in certain contexts

This is loaded from ``config/defaults/safety.yaml`` at startup and
can be extended at runtime via the SafetyValidator API.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety classification for an action."""

    SAFE = auto()
    """No confirmation required. Read text, describe scene, etc."""

    CAUTIOUS = auto()
    """Requires elevated confidence. Navigate, call, send message."""

    DANGEROUS = auto()
    """Requires voice confirmation. Emergency services, route to highway."""

    BLOCKED = auto()
    """Action is not allowed in the current context."""


@dataclass
class SafetyRule:
    """Safety rule for a single action.

    Attributes:
        action:             Action name (e.g. ``"navigate"``).
        level:              Base safety level.
        min_confidence:     Minimum STT confidence to accept (0.0–1.0).
        confirm_message:    Message to speak when requesting confirmation.
        dangerous_patterns: Regex patterns in arguments that elevate to DANGEROUS.
        blocked_patterns:   Regex patterns that block the action entirely.
        requires_context:   If True, action is blocked if context is missing.
    """

    action: str
    level: SafetyLevel = SafetyLevel.SAFE
    min_confidence: float = 0.50
    confirm_message: str = "Are you sure?"
    dangerous_patterns: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    requires_context: bool = False

    # Compiled patterns (populated lazily)
    _compiled_dangerous: Optional[List[re.Pattern]] = field(
        default=None, init=False, repr=False
    )
    _compiled_blocked: Optional[List[re.Pattern]] = field(
        default=None, init=False, repr=False
    )

    def get_dangerous_patterns(self) -> List[re.Pattern]:
        """Return compiled dangerous argument patterns."""
        if self._compiled_dangerous is None:
            self._compiled_dangerous = [
                re.compile(p, re.IGNORECASE) for p in self.dangerous_patterns
            ]
        return self._compiled_dangerous

    def get_blocked_patterns(self) -> List[re.Pattern]:
        """Return compiled blocked argument patterns."""
        if self._compiled_blocked is None:
            self._compiled_blocked = [
                re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
            ]
        return self._compiled_blocked


# Default safety rules for SIMON actions
DEFAULT_SAFETY_RULES: Dict[str, SafetyRule] = {
    # Navigation — cautious by default, dangerous for highways
    "navigate": SafetyRule(
        action="navigate",
        level=SafetyLevel.CAUTIOUS,
        min_confidence=0.70,
        confirm_message="Navigate to {destination}. Is that correct?",
        dangerous_patterns=[
            r"\bhighway\b", r"\bfreeway\b", r"\bmotorway\b",
            r"\bexpressway\b", r"\binterstate\b",
        ],
    ),
    # Emergency actions — always require confirmation
    "emergency_call": SafetyRule(
        action="emergency_call",
        level=SafetyLevel.DANGEROUS,
        min_confidence=0.85,
        confirm_message="Calling emergency services. Confirm?",
    ),
    "call": SafetyRule(
        action="call",
        level=SafetyLevel.CAUTIOUS,
        min_confidence=0.75,
        confirm_message="Call {contact}. Is that correct?",
        dangerous_patterns=[r"\b(911|112|999|100|108)\b"],
    ),
    # Reading/OCR — safe
    "read_text": SafetyRule(
        action="read_text",
        level=SafetyLevel.SAFE,
        min_confidence=0.50,
    ),
    "describe_scene": SafetyRule(
        action="describe_scene",
        level=SafetyLevel.SAFE,
        min_confidence=0.50,
    ),
    "read_sign": SafetyRule(
        action="read_sign",
        level=SafetyLevel.SAFE,
        min_confidence=0.50,
    ),
    # System commands
    "stop": SafetyRule(
        action="stop",
        level=SafetyLevel.SAFE,
        min_confidence=0.45,
    ),
    "help": SafetyRule(
        action="help",
        level=SafetyLevel.SAFE,
        min_confidence=0.40,
    ),
    "battery": SafetyRule(
        action="battery",
        level=SafetyLevel.SAFE,
        min_confidence=0.45,
    ),
    "time": SafetyRule(
        action="time",
        level=SafetyLevel.SAFE,
        min_confidence=0.40,
    ),
    "date": SafetyRule(
        action="date",
        level=SafetyLevel.SAFE,
        min_confidence=0.40,
    ),
}
