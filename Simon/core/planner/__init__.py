"""Task Planner — decomposes voice commands into executable task sequences.

Converts high-level user intents into ordered action sequences:
- "Navigate to Central Park" → [geocode, compute_route, start_navigation]
- "What do you see?" → [capture_frame, detect, describe_scene, speak]
- "Who is that?" → [capture_frame, detect_faces, identify, speak]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger("simon.core.planner")


class TaskStatus(IntEnum):
    """Task execution status."""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4


@dataclass
class Task:
    """A single executable task in a plan.

    Attributes
    ----------
    action : str
        Action identifier (e.g. ``"describe_scene"``, ``"navigate_to"``).
    params : dict
        Parameters for the action.
    status : TaskStatus
        Current execution status.
    result : Any
        Result from execution (set after completion).
    error : str
        Error message (set on failure).
    """

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    depends_on: Optional[str] = None  # previous task action to wait for

    @property
    def is_done(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def complete(self, result: Any = None) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def cancel(self) -> None:
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()


@dataclass
class TaskPlan:
    """An ordered sequence of tasks generated from a user intent.

    Attributes
    ----------
    intent : str
        The original user intent.
    tasks : list[Task]
        Ordered tasks to execute.
    """

    intent: str
    tasks: list[Task] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return all(t.is_done for t in self.tasks)

    @property
    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    @property
    def current_task(self) -> Optional[Task]:
        for task in self.tasks:
            if not task.is_done:
                return task
        return None

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        done = sum(1 for t in self.tasks if t.is_done)
        return done / len(self.tasks)


class TaskPlanner:
    """Decomposes user intents into executable task plans.

    Uses a strategy pattern: intent keywords map to plan-building
    functions that generate the correct task sequence.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Callable[[dict], TaskPlan]] = {}
        self._register_defaults()

    def register_strategy(
        self, intent: str, builder: Callable[[dict], TaskPlan]
    ) -> None:
        """Register a plan-building strategy for an intent."""
        self._strategies[intent] = builder

    def plan(self, intent: str, params: Optional[dict] = None) -> TaskPlan:
        """Create a task plan for a user intent.

        Parameters
        ----------
        intent : str
            User intent identifier (e.g. ``"navigate"``, ``"describe"``).
        params : dict, optional
            Parameters for the intent.

        Returns
        -------
        TaskPlan
        """
        params = params or {}
        builder = self._strategies.get(intent)

        if builder:
            return builder(params)

        # Default: single-task plan
        return TaskPlan(
            intent=intent,
            tasks=[Task(action=intent, params=params)],
        )

    def _register_defaults(self) -> None:
        """Register built-in planning strategies."""

        self._strategies["navigate"] = self._plan_navigate
        self._strategies["describe"] = self._plan_describe
        self._strategies["identify"] = self._plan_identify
        self._strategies["read_text"] = self._plan_read_text
        self._strategies["status"] = self._plan_status

    @staticmethod
    def _plan_navigate(params: dict) -> TaskPlan:
        destination = params.get("destination", "")
        tasks = [
            Task(action="geocode", params={"address": destination}),
            Task(action="compute_route", depends_on="geocode"),
            Task(action="start_navigation", depends_on="compute_route"),
            Task(action="speak", params={"text": f"Navigating to {destination}"}),
        ]
        return TaskPlan(intent="navigate", tasks=tasks)

    @staticmethod
    def _plan_describe(params: dict) -> TaskPlan:
        tasks = [
            Task(action="get_world_model"),
            Task(action="describe_scene", depends_on="get_world_model"),
            Task(action="speak", depends_on="describe_scene"),
        ]
        return TaskPlan(intent="describe", tasks=tasks)

    @staticmethod
    def _plan_identify(params: dict) -> TaskPlan:
        tasks = [
            Task(action="get_world_model"),
            Task(action="identify_faces", depends_on="get_world_model"),
            Task(action="speak", depends_on="identify_faces"),
        ]
        return TaskPlan(intent="identify", tasks=tasks)

    @staticmethod
    def _plan_read_text(params: dict) -> TaskPlan:
        tasks = [
            Task(action="get_world_model"),
            Task(action="read_ocr_text", depends_on="get_world_model"),
            Task(action="speak", depends_on="read_ocr_text"),
        ]
        return TaskPlan(intent="read_text", tasks=tasks)

    @staticmethod
    def _plan_status(params: dict) -> TaskPlan:
        tasks = [
            Task(action="get_system_status"),
            Task(action="speak", depends_on="get_system_status"),
        ]
        return TaskPlan(intent="status", tasks=tasks)
