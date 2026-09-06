"""Event base class and system event types.

Events are the primary inter-subsystem communication mechanism in SIMON.
Every event has a topic, priority, source identifier, correlation ID,
and an arbitrary data payload.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from core.models.enums import Priority


@dataclass
class Event:
    """Base event for the SIMON event bus.

    Parameters
    ----------
    topic : str
        Hierarchical topic, e.g. ``"vision.detection"``, ``"safety.hazard"``.
    data : dict[str, Any]
        Event-specific payload.
    priority : int
        Dispatch priority (1 = emergency, 10 = debug).
    source : str
        Originating subsystem name.
    correlation_id : str
        Trace ID for tracking events across subsystems.  Auto-generated.
    timestamp : float
        Event creation time.  Auto-set to ``time.time()``.
    is_safety_critical : bool
        If True, the event bus dispatches this synchronously (bypasses queue).
    """

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    priority: int = Priority.INFORMATIONAL
    source: str = "system"
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    is_safety_critical: bool = False

    def __post_init__(self) -> None:
        # Auto-flag safety events based on priority
        if self.priority <= Priority.SAFETY_CRITICAL:
            self.is_safety_critical = True

    def __lt__(self, other: Event) -> bool:
        """Priority queue ordering: lower priority value = higher urgency."""
        if not isinstance(other, Event):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def derive(self, topic: str, **overrides: Any) -> Event:
        """Create a new event derived from this one, preserving correlation_id.

        Useful for transforming events as they flow through the pipeline
        (e.g. ``vision.world_update`` → ``safety.hazard``).
        """
        return Event(
            topic=topic,
            data=overrides.pop("data", dict(self.data)),
            priority=overrides.pop("priority", self.priority),
            source=overrides.pop("source", self.source),
            correlation_id=self.correlation_id,
            is_safety_critical=overrides.pop(
                "is_safety_critical", self.is_safety_critical
            ),
        )
