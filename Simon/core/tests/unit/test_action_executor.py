"""Unit tests for ActionExecutor and ActionRegistry."""

from __future__ import annotations

import pytest

from core.events.event_bus import EventBus
from core.models.actions import Action, ActionResult
from core.models.enums import Priority
from core.actions.action_registry import ActionRegistry
from core.actions.action_executor import ActionExecutor
from core.planner.task import Task, TaskType


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def registry():
    return ActionRegistry()


@pytest.fixture
def executor(registry, event_bus):
    return ActionExecutor(action_registry=registry, event_bus=event_bus)


# ── ActionRegistry tests ────────────────────────────────────────────


class TestActionRegistry:
    def test_register_and_get(self, registry):
        handler = lambda a: ActionResult(action=a, success=True)
        registry.register("test_action", handler)
        assert registry.get_handler("test_action") is handler

    def test_get_missing_handler(self, registry):
        assert registry.get_handler("nonexistent") is None

    def test_has_handler(self, registry):
        handler = lambda a: ActionResult(action=a, success=True)
        registry.register("speak", handler)
        assert registry.has_handler("speak")
        assert not registry.has_handler("fly")

    def test_list_actions(self, registry):
        registry.register("a", lambda a: ActionResult(action=a, success=True))
        registry.register("b", lambda a: ActionResult(action=a, success=True))
        actions = registry.list_actions()
        assert "a" in actions
        assert "b" in actions

    def test_unregister(self, registry):
        handler = lambda a: ActionResult(action=a, success=True)
        registry.register("temp", handler)
        assert registry.unregister("temp") is True
        assert registry.get_handler("temp") is None

    def test_unregister_missing(self, registry):
        assert registry.unregister("nonexistent") is False


# ── ActionExecutor tests ────────────────────────────────────────────


class TestActionExecutor:
    def test_execute_registered_action(self, registry, executor):
        registry.register(
            "greet",
            lambda a: ActionResult(
                action=a, success=True, message="Hello!",
            ),
        )
        action = Action(action_type="greet")
        result = executor.execute(action)
        assert result.success is True
        assert result.message == "Hello!"

    def test_execute_unregistered_action(self, executor):
        action = Action(action_type="unknown_thing")
        result = executor.execute(action)
        assert result.success is False
        assert "No handler" in result.error

    def test_execute_handler_exception(self, registry, executor):
        def bad_handler(a):
            raise ValueError("something broke")

        registry.register("fail", bad_handler)
        action = Action(action_type="fail")
        result = executor.execute(action)
        assert result.success is False
        assert "something broke" in result.error

    def test_execute_records_duration(self, registry, executor):
        registry.register(
            "slow",
            lambda a: ActionResult(action=a, success=True),
        )
        action = Action(action_type="slow")
        result = executor.execute(action)
        assert result.duration_ms >= 0

    def test_execute_task_sequential(self, registry, executor):
        """Execute a task with multiple subtasks sequentially."""
        calls = []

        def handler(a):
            calls.append(a.action_type)
            return ActionResult(action=a, success=True)

        registry.register("speak", handler)
        registry.register("navigate", handler)

        task = Task(
            task_type=TaskType.CUSTOM,
            subtasks=[
                Task(task_type=TaskType.SPEAK, args={"text": "Going"}),
                Task(task_type=TaskType.NAVIGATE, args={"destination": "park"}),
            ],
        )
        result = executor.execute_task(task)
        assert result.success is True
        assert calls == ["speak", "navigate"]
        assert task.status.name == "COMPLETED"

    def test_execute_task_stops_on_failure(self, registry, executor):
        """Subtask failure should stop execution."""
        registry.register(
            "speak",
            lambda a: ActionResult(action=a, success=False, error="TTS down"),
        )
        registry.register(
            "navigate",
            lambda a: ActionResult(action=a, success=True),
        )

        task = Task(
            task_type=TaskType.CUSTOM,
            subtasks=[
                Task(task_type=TaskType.SPEAK),
                Task(task_type=TaskType.NAVIGATE),
            ],
        )
        result = executor.execute_task(task)
        assert result.success is False
        assert task.status.name == "FAILED"
        # Second subtask should still be pending
        assert task.subtasks[1].status.name == "PENDING"
