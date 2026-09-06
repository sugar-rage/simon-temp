"""
Speech priority levels — defines the preemption hierarchy for the TTS queue.

Lower numeric value == higher urgency.  The priority queue uses these
values to decide whether incoming speech should interrupt currently-playing
audio.  This is critical for safety: an emergency obstacle warning must
pre-empt a leisurely OCR reading.

Mapped from the existing priority system in speech_output.py (1 = highest)
and config.py (VOICE_PRIORITY_OBSTACLE = 1, etc.), but expanded with
finer granularity.
"""

from __future__ import annotations

from enum import IntEnum


class SpeechPriority(IntEnum):
    """Priority levels for the speech output queue.

    Lower value == higher priority == more urgent.
    Used by :class:`PriorityQueue` to order and pre-empt speech.

    The numeric values are intentionally sparse to allow future insertion
    of intermediate priority levels without renumbering.
    """

    # ------ Safety-critical (always interrupt) ------ #
    EMERGENCY = 1
    """Imminent collision, fall detection, system failure.
    Must interrupt ALL other speech immediately."""

    OBSTACLE = 2
    """Nearby obstacle warning (car, bicycle, etc.).
    Interrupts everything except EMERGENCY."""

    # ------ Navigation ------ #
    NAVIGATION = 3
    """Turn-by-turn instructions, arrival notices, re-routing.
    Interrupts lower-priority speech."""

    # ------ Perception ------ #
    OCR = 4
    """Read-aloud text from OCR scanning.
    Important but not urgent."""

    FACE = 5
    """Face recognition announcements.
    Informational."""

    # ------ Informational ------ #
    STATUS = 6
    """System status responses ("I see 3 objects, navigation idle")."""

    NOTIFICATION = 7
    """General notifications, confirmations, non-critical feedback."""

    INFO = 8
    """Low-priority information, tips, background updates."""

    # ------ Internal ------ #
    CONFIRMATION = 9
    """Safety confirmation prompts ("Did you mean navigate to highway?").
    Slightly lower than the command that triggered it, to avoid
    self-interruption."""

    DEBUG = 10
    """Debug/diagnostic messages.  Never played in production."""

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @property
    def should_interrupt(self) -> bool:
        """True if this priority level should interrupt lower-priority speech."""
        return self.value <= self.NAVIGATION.value

    @property
    def is_safety_critical(self) -> bool:
        """True if this is a safety-critical priority (EMERGENCY or OBSTACLE)."""
        return self.value <= self.OBSTACLE.value

    @property
    def label(self) -> str:
        """Human-readable label for logging."""
        return self.name.replace("_", " ").title()

    @classmethod
    def from_legacy(cls, priority_int: int) -> SpeechPriority:
        """Convert legacy integer priority (1–10) to SpeechPriority.

        The existing codebase uses bare integers (1 = highest, 5 = default).
        This bridges old code during migration.
        """
        mapping = {
            1: cls.EMERGENCY,
            2: cls.OBSTACLE,
            3: cls.NAVIGATION,
            4: cls.OCR,
            5: cls.FACE,
            6: cls.STATUS,
            7: cls.NOTIFICATION,
            8: cls.INFO,
            9: cls.CONFIRMATION,
            10: cls.DEBUG,
        }
        return mapping.get(priority_int, cls.INFO)
