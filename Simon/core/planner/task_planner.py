"""Task Planner — decomposes voice commands into executable task trees.

Receives speech commands (action + args), applies a planning strategy
to decompose them into ordered subtasks, and tracks task lifecycle.

The TaskPlanner owns the active-task registry and publishes
``logic.task_complete`` events on completion or failure.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority, TaskStatus
from core.planner.task import Task, TaskType
from core.planner.strategies import PlanningStrategy, get_default_strategies

logger = logging.getLogger("simon.core.planner")


class TaskPlanner:
    """Decomposes voice commands into executable task trees.

    Parameters
    ----------
    event_bus : EventBus
        For publishing task lifecycle events.
    strategies : dict[str, PlanningStrategy], optional
        Custom strategy map.  Defaults to built-in strategies.
    max_active_tasks : int
        Maximum concurrent active tasks.
    """

    def __init__(
        self,
        event_bus: EventBus,
        strategies: Optional[dict[str, PlanningStrategy]] = None,
        max_active_tasks: int = 10,
    ) -> None:
        self._event_bus = event_bus
        self._strategies = strategies or get_default_strategies()
        self._max_active = max_active_tasks
        self._active_tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────

    def plan_command(self, action: str, args: dict[str, Any] | str = "") -> Optional[Task]:
        """Decompose a voice command into an executable task tree.

        Parameters
        ----------
        action : str
            Command action string (e.g. ``"navigate"``, ``"status"``).
        args : dict or str
            Command arguments.  If a string, wrapped as ``{"args": value}``.

        Returns
        -------
        Task or None
            Root task with subtasks, or None if no strategy found.
        """
        # Normalize args
        if isinstance(args, str):
            args = {"args": args} if args else {}

        strategy = self._strategies.get(action)
        if strategy is None:
            logger.warning("No planning strategy for action=%r", action)
            return None

        # Build subtasks
        subtasks = strategy.plan(args)

        # Create root task wrapping the subtasks
        root = Task(
            task_type=TaskType.CUSTOM if action not in TaskType.__members__ else TaskType(action),
            args=args,
            subtasks=subtasks,
            priority=subtasks[0].priority if subtasks else Priority.INFORMATIONAL,
        )

        # Set parent IDs on subtasks
        for sub in subtasks:
            sub.parent_id = root.task_id

        # Register as active
        with self._lock:
            self._enforce_limit()
            self._active_tasks[root.task_id] = root

        logger.info(
            "Planned task %s: action=%r, subtasks=%d",
            root.task_id, action, len(subtasks),
        )
        return root

    def get_active_tasks(self) -> list[Task]:
        """Return all active (pending or running) tasks.

        Returns
        -------
        list[Task]
            Snapshot of active tasks (safe to iterate).
        """
        with self._lock:
            return [t for t in self._active_tasks.values() if t.is_active]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by ID.

        Parameters
        ----------
        task_id : str
            ID of the task to cancel.

        Returns
        -------
        bool
            True if the task was found and cancelled.
        """
        with self._lock:
            task = self._active_tasks.get(task_id)
            if task is None:
                return False
            task.cancel()
            for sub in task.subtasks:
                if sub.is_active:
                    sub.cancel()

        logger.info("Cancelled task %s", task_id)
        self._publish_completion(task)
        return True

    def cancel_all(self) -> int:
        """Cancel all active tasks.

        Returns
        -------
        int
            Number of tasks cancelled.
        """
        cancelled = 0
        with self._lock:
            for task in list(self._active_tasks.values()):
                if task.is_active:
                    task.cancel()
                    for sub in task.subtasks:
                        if sub.is_active:
                            sub.cancel()
                    cancelled += 1

        if cancelled:
            logger.info("Cancelled %d active tasks", cancelled)
        return cancelled

    def on_task_complete(self, task_id: str, result: Optional[dict] = None) -> None:
        """Mark a task as completed.

        Parameters
        ----------
        task_id : str
            ID of the completed task.
        result : dict, optional
            Task result data.
        """
        with self._lock:
            task = self._active_tasks.get(task_id)
            if task is None:
                return
            task.complete(result)

        logger.info("Task %s completed", task_id)
        self._publish_completion(task)

    def on_task_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed.

        Parameters
        ----------
        task_id : str
            ID of the failed task.
        error : str
            Error description.
        """
        with self._lock:
            task = self._active_tasks.get(task_id)
            if task is None:
                return
            task.fail(error)

        logger.warning("Task %s failed: %s", task_id, error)
        self._publish_completion(task)

    def register_strategy(self, action: str, strategy: PlanningStrategy) -> None:
        """Register a custom planning strategy for an action.

        Parameters
        ----------
        action : str
            Command action string.
        strategy : PlanningStrategy
            Strategy to use for this action.
        """
        self._strategies[action] = strategy
        logger.info("Registered strategy for action=%r", action)

    # ── Internal ─────────────────────────────────────────────────────

    def _enforce_limit(self) -> None:
        """Remove terminal tasks if we're over the active limit."""
        # Prune terminal tasks first
        terminal_ids = [
            tid for tid, t in self._active_tasks.items() if t.is_terminal
        ]
        for tid in terminal_ids:
            del self._active_tasks[tid]

        # If still over limit, remove oldest terminal
        if len(self._active_tasks) >= self._max_active:
            oldest_id = min(
                self._active_tasks,
                key=lambda tid: self._active_tasks[tid].created_at,
            )
            del self._active_tasks[oldest_id]
            logger.debug("Evicted oldest task %s due to limit", oldest_id)

    def _publish_completion(self, task: Task) -> None:
        """Publish task completion event."""
        try:
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_TASK_COMPLETE,
                priority=Priority.INFORMATIONAL,
                source="task_planner",
                data={
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "status": task.status.name,
                    "error": task.error,
                    "duration_ms": task.duration_ms,
                },
            ))
        except Exception:
            logger.error("Failed to publish task completion event", exc_info=True)
