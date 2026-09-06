"""Shared enumerations for SIMON subsystems."""

from __future__ import annotations

from enum import IntEnum, Enum, auto


class SubsystemStatus(Enum):
    """Lifecycle status of a subsystem."""

    UNKNOWN = auto()
    UNINITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class Priority(IntEnum):
    """Universal priority levels (lower number = higher priority).

    Used for event dispatch ordering, speech queue ordering, and
    announcement scheduling.
    """

    EMERGENCY = 1
    SAFETY_CRITICAL = 2
    OBSTACLE = 3
    NAVIGATION = 4
    FACE = 5
    INFORMATIONAL = 6
    AMBIENT = 7
    LOW = 8
    BACKGROUND = 9
    DEBUG = 10

    @property
    def is_safety_critical(self) -> bool:
        """Return True for priorities that should bypass normal queuing."""
        return self.value <= Priority.SAFETY_CRITICAL

    @property
    def should_interrupt(self) -> bool:
        """Return True for priorities that should interrupt current speech."""
        return self.value <= Priority.OBSTACLE


class DetectionSource(Enum):
    """Source of a detection result."""

    YOLO = "yolo"
    FACE = "face"
    OCR = "ocr"
    SCENE = "scene"
    DEPTH = "depth"


class HazardLevel(IntEnum):
    """Hazard severity classification.

    Level 1 — CRITICAL: Moving vehicle <2m, approaching fast.
    Level 2 — WARNING: Vehicle/obstacle <3m.
    Level 3 — CAUTION: Object detected, not urgent.
    Level 4 — INFO: Informational (face, text, navigation).
    """

    CRITICAL = 1
    WARNING = 2
    CAUTION = 3
    INFO = 4

    @property
    def is_urgent(self) -> bool:
        return self.value <= HazardLevel.WARNING


class NavigationManeuver(Enum):
    """Turn-by-turn navigation maneuver types."""

    STRAIGHT = "straight"
    TURN_LEFT = "turn-left"
    TURN_RIGHT = "turn-right"
    SLIGHT_LEFT = "slight-left"
    SLIGHT_RIGHT = "slight-right"
    SHARP_LEFT = "sharp-left"
    SHARP_RIGHT = "sharp-right"
    U_TURN = "u-turn"
    ARRIVE = "arrive"
    DEPART = "depart"


class TaskStatus(Enum):
    """Lifecycle state of a planned task."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
