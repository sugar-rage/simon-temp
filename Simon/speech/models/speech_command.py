"""
Speech command data model — the output of the listen pipeline.

Replaces the legacy ``{"action": str, "args": str}`` dict with a typed
dataclass that carries confidence, safety classification, raw transcript,
and provenance metadata.  The ``action`` and ``args`` properties maintain
backward compatibility with existing code in ``main.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class CommandSource(Enum):
    """How the command was produced."""
    VOICE = auto()
    KEYBOARD = auto()
    INTERNAL = auto()   # System-generated (e.g. context resolution)


class CommandStatus(Enum):
    """Lifecycle status of a command."""
    PENDING = auto()            # Parsed, awaiting safety validation
    APPROVED = auto()           # Passed all gates, ready to execute
    NEEDS_CLARIFICATION = auto()  # Below confidence threshold
    NEEDS_CONFIRMATION = auto()   # Safety rule requires confirmation
    BLOCKED = auto()            # Failed safety validation
    EXECUTED = auto()           # Successfully executed
    EXPIRED = auto()            # Timed out before execution


# Safety-critical actions that require elevated confidence thresholds
# and may trigger the confirmation dialog.
SAFETY_CRITICAL_ACTIONS = frozenset({
    "navigate",
    "cancel_nav",
    "save_face",
    "stop",
})

# Actions that change navigation state and must NEVER be executed
# with uncertain recognition.
NAVIGATION_ACTIONS = frozenset({
    "navigate",
    "cancel_nav",
})


@dataclass
class SpeechCommand:
    """A parsed voice command with confidence and safety metadata.

    Attributes:
        action:           Normalised action name (e.g. "navigate", "read_text").
        args:             Extracted arguments (e.g. destination name).
        confidence:       ASR confidence score [0.0, 1.0] for this transcript.
        raw_transcript:   The unprocessed text from the STT engine.
        source:           How the command was created (voice, keyboard, etc.).
        status:           Current lifecycle status.
        timestamp:        When the command was created (monotonic time).
        engine_name:      Name of the STT engine that produced this transcript.
        decoding_passes:  Number of decoding attempts before settling.
        speaker_verified: Whether the speaker passed verification (None = skipped).
    """

    action: str
    args: str = ""
    confidence: float = 0.0
    raw_transcript: str = ""
    source: CommandSource = CommandSource.VOICE
    status: CommandStatus = CommandStatus.PENDING
    timestamp: float = field(default_factory=time.monotonic)
    engine_name: str = ""
    decoding_passes: int = 1
    speaker_verified: Optional[bool] = None

    # ------------------------------------------------------------------ #
    #  Safety classification
    # ------------------------------------------------------------------ #

    @property
    def is_safety_critical(self) -> bool:
        """True if this action is safety-critical and requires elevated confidence."""
        return self.action in SAFETY_CRITICAL_ACTIONS

    @property
    def is_navigation(self) -> bool:
        """True if this action changes navigation state."""
        return self.action in NAVIGATION_ACTIONS

    # ------------------------------------------------------------------ #
    #  Backward compatibility with legacy dict format
    # ------------------------------------------------------------------ #

    def to_legacy_dict(self) -> dict:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "action": self.action,
            "args": self.args,
            "raw_transcript": self.raw_transcript,
            "confidence": self.confidence,
        }

    def __getitem__(self, key: str):
        """Support dict-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default=None):
        """Support dict-style .get() for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

    # ------------------------------------------------------------------ #
    #  Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        parts = [f"action={self.action!r}"]
        if self.args:
            parts.append(f"args={self.args!r}")
        parts.append(f"conf={self.confidence:.2f}")
        parts.append(f"status={self.status.name}")
        if self.engine_name:
            parts.append(f"engine={self.engine_name}")
        return f"SpeechCommand({', '.join(parts)})"
