"""Action Executor — dispatches actions to registered handlers.

The executor is the bridge between the TaskPlanner (which produces
abstract ``Action`` objects) and the subsystem handlers (which do the
actual work).

Metrics:
- ``core.actions_executed`` — counter of all executed actions
- ``core.action_latency_ms`` — histogram of execution times
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.actions import Action, ActionResult
from core.models.events import Event
from core.models.enums import Priority
from core.actions.action_registry import ActionRegistry
from core.metrics.collector import SystemMetricsCollector
from core.planner.task import Task

logger = logging.getLogger("simon.core.actions")


class ActionExecutor:
    """Dispatches actions to registered handlers.

    Parameters
    ----------
    action_registry : ActionRegistry
        Registry of action type → handler mappings.
    event_bus : EventBus
        For publishing decision events.
    """

    def __init__(
        self,
        action_registry: ActionRegistry,
        event_bus: EventBus,
    ) -> None:
        self._registry = action_registry
        self._event_bus = event_bus
        self._metrics = SystemMetricsCollector.get()

    def execute(self, action: Action) -> ActionResult:
        """Execute a single action.

        Looks up the handler in the registry, invokes it, records
        metrics, and publishes a ``logic.decision`` event.

        Parameters
        ----------
        action : Action
            The action to execute.

        Returns
        -------
        ActionResult
            Execution result.
        """
        t0 = time.time()
        handler = self._registry.get_handler(action.action_type)

        if handler is None:
            elapsed = (time.time() - t0) * 1000
            result = ActionResult(
                action=action,
                success=False,
                error=f"No handler for action type: {action.action_type!r}",
                duration_ms=elapsed,
            )
            logger.warning("No handler for action_type=%r", action.action_type)
            self._publish_decision(action, result)
            return result

        try:
            result = handler(action)
            # Ensure duration is set
            if result.duration_ms <= 0:
                result.duration_ms = (time.time() - t0) * 1000
        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            result = ActionResult(
                action=action,
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )
            logger.error(
                "Handler for action_type=%r raised %s: %s",
                action.action_type, type(exc).__name__, exc,
                exc_info=True,
            )

        # Record metrics
        self._metrics.increment("core.actions_executed")
        self._metrics.record_time("core.action_latency_ms", result.duration_ms)

        # Publish decision event
        self._publish_decision(action, result)

        level = logging.INFO if result.success else logging.WARNING
        logger.log(
            level,
            "Executed action=%r success=%s duration=%.1fms",
            action.action_type, result.success, result.duration_ms,
        )
        return result

    def execute_task(self, task: Task) -> ActionResult:
        """Execute a task's subtasks sequentially.

        Runs each subtask in order.  Stops on first failure unless
        the failing subtask is non-critical.

        Parameters
        ----------
        task : Task
            Root task with subtasks to execute.

        Returns
        -------
        ActionResult
            Result of the last executed subtask, or a failure result
            if a subtask failed.
        """
        task.start()
        last_result: Optional[ActionResult] = None

        subtasks = task.subtasks if task.subtasks else [task]

        for sub in subtasks:
            if not task.is_active:
                # Task was cancelled externally
                break

            sub.start()
            action = Action(
                action_type=sub.task_type.value,
                args=sub.args,
                priority=sub.priority,
                correlation_id=task.task_id,
            )
            result = self.execute(action)
            last_result = result

            if result.success:
                sub.complete(result.data)
            else:
                sub.fail(result.error or "Unknown error")
                task.fail(result.error or f"Subtask {sub.task_type.value} failed")
                return result

        # All subtasks completed successfully
        if task.is_active:
            task.complete(last_result.data if last_result else {})

        return last_result or ActionResult(
            action=Action(action_type="noop"),
            success=True,
            message="No subtasks",
        )

    def _publish_decision(self, action: Action, result: ActionResult) -> None:
        """Publish a logic.decision event."""
        try:
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_DECISION,
                priority=Priority.LOW,
                source="action_executor",
                data={
                    "action_type": action.action_type,
                    "success": result.success,
                    "message": result.message,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                },
                correlation_id=action.correlation_id or "",
            ))
        except Exception:
            logger.error("Failed to publish decision event", exc_info=True)
