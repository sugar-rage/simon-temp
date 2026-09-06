"""Planning strategies for task decomposition.

Each strategy knows how to decompose a specific type of voice command
into an ordered list of subtasks.

Design decision: strategies are pure functions wrapped in classes
for testability and extensibility (new strategies for new command types).
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from core.planner.task import Task, TaskType
from core.models.enums import Priority

logger = logging.getLogger("simon.core.planner")


class PlanningStrategy(abc.ABC):
    """Abstract strategy for decomposing a command into subtasks."""

    @abc.abstractmethod
    def plan(self, args: dict[str, Any]) -> list[Task]:
        """Decompose a command into ordered subtasks.

        Parameters
        ----------
        args : dict
            Command arguments (e.g. ``{"destination": "library"}``).

        Returns
        -------
        list[Task]
            Ordered list of subtasks to execute.
        """


class SimpleStrategy(PlanningStrategy):
    """Passthrough strategy for single-action commands.

    Used for commands that don't need decomposition: status, stop,
    read_text, save_face, describe_scene, cancel_nav.
    """

    def __init__(self, task_type: TaskType, priority: int = Priority.INFORMATIONAL) -> None:
        self._task_type = task_type
        self._priority = priority

    def plan(self, args: dict[str, Any]) -> list[Task]:
        """Return a single subtask matching the command."""
        return [
            Task(
                task_type=self._task_type,
                args=args,
                priority=self._priority,
            )
        ]


class NavigationStrategy(PlanningStrategy):
    """Multi-step decomposition for navigation commands.

    Produces::

        SubTask 1: SPEAK — "Finding route to <destination>"
        SubTask 2: NAVIGATE — trigger route computation + guidance
    """

    def plan(self, args: dict[str, Any]) -> list[Task]:
        """Decompose a navigation command into subtasks."""
        destination = args.get("destination", args.get("args", ""))
        if not destination:
            # No destination — produce a single speak task asking for it
            return [
                Task(
                    task_type=TaskType.SPEAK,
                    args={"text": "Where would you like to go?", "priority": Priority.INFORMATIONAL},
                    priority=Priority.INFORMATIONAL,
                )
            ]

        return [
            Task(
                task_type=TaskType.SPEAK,
                args={
                    "text": f"Finding route to {destination}",
                    "priority": Priority.NAVIGATION,
                },
                priority=Priority.NAVIGATION,
            ),
            Task(
                task_type=TaskType.NAVIGATE,
                args={"destination": destination},
                priority=Priority.NAVIGATION,
            ),
        ]


# ── Default strategy registry ───────────────────────────────────────

def get_default_strategies() -> dict[str, PlanningStrategy]:
    """Return the built-in strategy map for all known command types.

    Returns
    -------
    dict[str, PlanningStrategy]
        Maps voice command action strings to their planning strategies.
    """
    return {
        "navigate": NavigationStrategy(),
        "read_text": SimpleStrategy(TaskType.READ_TEXT, Priority.INFORMATIONAL),
        "save_face": SimpleStrategy(TaskType.SAVE_FACE, Priority.INFORMATIONAL),
        "describe_scene": SimpleStrategy(TaskType.DESCRIBE_SCENE, Priority.INFORMATIONAL),
        "status": SimpleStrategy(TaskType.REPORT_STATUS, Priority.LOW),
        "stop": SimpleStrategy(TaskType.STOP, Priority.INFORMATIONAL),
        "cancel_nav": SimpleStrategy(TaskType.CANCEL_NAV, Priority.NAVIGATION),
    }
