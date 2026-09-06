"""Action system — registry-based command execution.

Provides:
- ActionHandler: ABC for all action implementations
- ActionRegistry: maps action names to handlers
- ActionExecutor: executes TaskPlans by dispatching actions
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Callable, Optional

from core.planner import Task, TaskPlan, TaskStatus

logger = logging.getLogger("simon.core.actions")


class ActionHandler(abc.ABC):
    """Abstract base for action handlers."""

    @abc.abstractmethod
    def execute(self, params: dict[str, Any]) -> Any:
        """Execute the action with given parameters.

        Returns the action result, or raises on failure.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ActionRegistry:
    """Registry mapping action names to handler callables.

    Supports both ActionHandler instances and plain functions.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict], Any]] = {}

    def register(self, action: str, handler: Callable[[dict], Any]) -> None:
        """Register a handler for an action name."""
        self._handlers[action] = handler
        logger.debug("Registered action: %s", action)

    def register_handler(self, action: str, handler: ActionHandler) -> None:
        """Register an ActionHandler instance."""
        self._handlers[action] = handler.execute

    def get(self, action: str) -> Optional[Callable[[dict], Any]]:
        """Get the handler for an action, or None."""
        return self._handlers.get(action)

    def has(self, action: str) -> bool:
        return action in self._handlers

    @property
    def action_names(self) -> list[str]:
        return list(self._handlers.keys())


class ActionExecutor:
    """Executes TaskPlans by dispatching tasks to registered action handlers.

    Parameters
    ----------
    registry : ActionRegistry
        Action handler registry.
    """

    def __init__(self, registry: ActionRegistry) -> None:
        self._registry = registry
        self._completed_plans: list[TaskPlan] = []

    def execute_plan(self, plan: TaskPlan) -> TaskPlan:
        """Execute all tasks in a plan sequentially.

        Tasks are executed in order. If a task fails and is not
        recoverable, subsequent tasks are cancelled.

        Returns the completed plan.
        """
        logger.info("Executing plan: %s (%d tasks)", plan.intent, len(plan.tasks))

        previous_results: dict[str, Any] = {}

        for task in plan.tasks:
            if task.is_done:
                continue

            # Check dependency
            if task.depends_on and task.depends_on not in previous_results:
                dep_tasks = [t for t in plan.tasks if t.action == task.depends_on]
                if dep_tasks and dep_tasks[0].status == TaskStatus.FAILED:
                    task.cancel()
                    continue

            task.status = TaskStatus.RUNNING

            handler = self._registry.get(task.action)
            if handler is None:
                task.fail(f"No handler registered for action: {task.action}")
                logger.warning("No handler for action: %s", task.action)
                continue

            try:
                # Inject previous task results into params
                params = dict(task.params)
                if task.depends_on and task.depends_on in previous_results:
                    params["_previous_result"] = previous_results[task.depends_on]

                result = handler(params)
                task.complete(result)
                previous_results[task.action] = result
                logger.debug("Task completed: %s", task.action)

            except Exception as e:
                task.fail(str(e))
                logger.error("Task failed: %s — %s", task.action, e)

        self._completed_plans.append(plan)
        return plan

    @property
    def completed_plan_count(self) -> int:
        return len(self._completed_plans)
