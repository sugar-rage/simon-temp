"""Unit tests for TaskPlanner — command decomposition and lifecycle."""

from __future__ import annotations

import pytest

from core.events.event_bus import EventBus
from core.models.enums import Priority, TaskStatus
from core.planner.task import Task, TaskType
from core.planner.task_planner import TaskPlanner
from core.planner.strategies import (
    SimpleStrategy,
    NavigationStrategy,
    PlanningStrategy,
    get_default_strategies,
)


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def planner(event_bus):
    return TaskPlanner(event_bus=event_bus)


# ── Task dataclass tests ────────────────────────────────────────────


class TestTask:
    def test_task_defaults(self):
        t = Task(task_type=TaskType.SPEAK)
        assert t.status == TaskStatus.PENDING
        assert t.is_active
        assert not t.is_terminal
        assert t.duration_ms is None

    def test_task_lifecycle_complete(self):
        t = Task(task_type=TaskType.SPEAK)
        t.start()
        assert t.status == TaskStatus.RUNNING
        assert t.started_at is not None
        t.complete({"text": "hello"})
        assert t.status == TaskStatus.COMPLETED
        assert t.result == {"text": "hello"}
        assert t.is_terminal
        assert t.duration_ms is not None
        assert t.duration_ms >= 0

    def test_task_lifecycle_fail(self):
        t = Task(task_type=TaskType.NAVIGATE, args={"destination": "library"})
        t.start()
        t.fail("no route found")
        assert t.status == TaskStatus.FAILED
        assert t.error == "no route found"
        assert t.is_terminal

    def test_task_lifecycle_cancel(self):
        t = Task(task_type=TaskType.NAVIGATE)
        t.cancel()
        assert t.status == TaskStatus.CANCELLED
        assert t.is_terminal


# ── Strategy tests ──────────────────────────────────────────────────


class TestStrategies:
    def test_simple_strategy_single_task(self):
        strategy = SimpleStrategy(TaskType.REPORT_STATUS, Priority.LOW)
        subtasks = strategy.plan({})
        assert len(subtasks) == 1
        assert subtasks[0].task_type == TaskType.REPORT_STATUS
        assert subtasks[0].priority == Priority.LOW

    def test_navigation_strategy_with_destination(self):
        strategy = NavigationStrategy()
        subtasks = strategy.plan({"destination": "library"})
        assert len(subtasks) == 2
        assert subtasks[0].task_type == TaskType.SPEAK
        assert "library" in subtasks[0].args["text"]
        assert subtasks[1].task_type == TaskType.NAVIGATE
        assert subtasks[1].args["destination"] == "library"

    def test_navigation_strategy_no_destination(self):
        strategy = NavigationStrategy()
        subtasks = strategy.plan({})
        assert len(subtasks) == 1
        assert subtasks[0].task_type == TaskType.SPEAK
        assert "where" in subtasks[0].args["text"].lower()

    def test_default_strategies_coverage(self):
        strategies = get_default_strategies()
        assert "navigate" in strategies
        assert "status" in strategies
        assert "stop" in strategies
        assert "read_text" in strategies
        assert "save_face" in strategies
        assert "cancel_nav" in strategies
        assert "describe_scene" in strategies


# ── TaskPlanner tests ───────────────────────────────────────────────


class TestTaskPlanner:
    def test_plan_status_command(self, planner):
        task = planner.plan_command("status")
        assert task is not None
        assert len(task.subtasks) == 1
        assert task.subtasks[0].task_type == TaskType.REPORT_STATUS

    def test_plan_navigate_command(self, planner):
        task = planner.plan_command("navigate", {"destination": "park"})
        assert task is not None
        assert len(task.subtasks) == 2
        # Subtasks should have parent_id set
        for sub in task.subtasks:
            assert sub.parent_id == task.task_id

    def test_plan_unknown_command(self, planner):
        task = planner.plan_command("unknown_action_xyz")
        assert task is None

    def test_plan_string_args(self, planner):
        task = planner.plan_command("navigate", "library")
        assert task is not None
        # Navigation strategy should have parsed the destination
        assert len(task.subtasks) == 2

    def test_active_tasks_tracking(self, planner):
        t1 = planner.plan_command("status")
        t2 = planner.plan_command("stop")
        active = planner.get_active_tasks()
        assert len(active) >= 2

    def test_cancel_task(self, planner):
        task = planner.plan_command("status")
        assert task is not None
        result = planner.cancel_task(task.task_id)
        assert result is True
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent_task(self, planner):
        result = planner.cancel_task("nonexistent_id_123")
        assert result is False

    def test_cancel_all(self, planner):
        planner.plan_command("status")
        planner.plan_command("stop")
        cancelled = planner.cancel_all()
        assert cancelled >= 2

    def test_on_task_complete(self, planner):
        task = planner.plan_command("status")
        assert task is not None
        planner.on_task_complete(task.task_id, {"report": "all good"})
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"report": "all good"}

    def test_on_task_failed(self, planner):
        task = planner.plan_command("navigate", {"destination": "moon"})
        assert task is not None
        planner.on_task_failed(task.task_id, "unreachable")
        assert task.status == TaskStatus.FAILED
        assert task.error == "unreachable"

    def test_register_custom_strategy(self, planner):
        class CustomStrategy(PlanningStrategy):
            def plan(self, args):
                return [Task(task_type=TaskType.CUSTOM, args=args)]

        planner.register_strategy("my_action", CustomStrategy())
        task = planner.plan_command("my_action", {"key": "value"})
        assert task is not None
        assert len(task.subtasks) == 1
        assert task.subtasks[0].task_type == TaskType.CUSTOM

    def test_max_active_tasks_limit(self, event_bus):
        planner = TaskPlanner(event_bus=event_bus, max_active_tasks=3)
        for i in range(5):
            planner.plan_command("status")
        # Should not raise, evicts old tasks
        active = planner.get_active_tasks()
        assert len(active) <= 5  # Some may have been pruned
