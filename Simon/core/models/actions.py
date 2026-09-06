"""Action and ActionResult dataclasses for the action executor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Action:
    """An abstract action to be executed by the ActionExecutor.

    Actions are produced by the TaskPlanner or voice command parser
    and dispatched to concrete handlers by the ActionExecutor.
    """

    action_type: str
    """Action identifier, e.g. ``"speak"``, ``"navigate"``, ``"read_text"``."""

    args: dict[str, Any] = field(default_factory=dict)
    """Action-specific arguments."""

    priority: int = 6
    """Execution priority (lower = more urgent)."""

    source: str = "user"
    """Who requested this action (``"user"``, ``"system"``, ``"safety"``)."""

    correlation_id: Optional[str] = None
    """Trace ID linking this action to its triggering event."""

    created_at: float = field(default_factory=time.time)


@dataclass
class ActionResult:
    """Result of executing an action."""

    action: Action
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    completed_at: float = field(default_factory=time.time)
