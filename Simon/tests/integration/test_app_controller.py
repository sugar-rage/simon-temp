"""Integration tests for AppController — end-to-end command flow.

These tests use mock subsystems (no hardware/ML required) to verify
that the AppController correctly wires dependencies, dispatches
commands, and manages lifecycle.
"""

from __future__ import annotations

import time
import pytest

from core.config.system_config import SystemConfig
from core.events.event_bus import EventBus
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.resources.model_registry import ModelRegistry
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor
from core.actions.action_registry import ActionRegistry
from core.actions.action_executor import ActionExecutor
from core.actions.action_handlers import register_builtin_handlers
from core.planner.task_planner import TaskPlanner
from core.models.enums import Priority


# We can't import AppController directly because it tries to import
# subsystems.  Instead, we test the core flow manually.


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def state_machine():
    sm = StateMachine(initial_state=SystemState.STARTING)
    sm.transition(SystemState.INITIALIZING)
    sm.transition(SystemState.READY)
    return sm


@pytest.fixture
def capabilities():
    reg = CapabilityRegistry()
    reg.register("capability.camera", available=True)
    reg.register("capability.detection", available=True)
    reg.register("capability.gps", available=True)
    reg.register("capability.speech", available=True)
    return reg


class MockSpeechManager:
    def __init__(self):
        self.spoken = []

    def speak(self, text, priority=6):
        self.spoken.append((text, priority))

    def start(self):
        pass

    def stop(self):
        pass


class MockNavPipeline:
    def __init__(self):
        self.navigating = False
        self.destination = None

    def navigate_to(self, dest):
        self.destination = dest
        self.navigating = True

    def cancel(self):
        self.navigating = False


class TestIntegrationCommandFlow:
    """Test the planner → executor → handler flow end-to-end."""

    def test_status_command_flow(self, event_bus, state_machine, capabilities):
        speech = MockSpeechManager()
        model_reg = ModelRegistry()
        watchdog = Watchdog(event_bus=event_bus, interval_s=60, timeout_s=120)
        health = HealthMonitor(
            capabilities=capabilities,
            watchdog=watchdog,
            model_registry=model_reg,
            state_machine=state_machine,
            event_bus=event_bus,
        )

        registry = ActionRegistry()
        register_builtin_handlers(
            registry=registry,
            speech_manager=speech,
            capabilities=capabilities,
            health_monitor=health,
            state_machine=state_machine,
        )

        planner = TaskPlanner(event_bus=event_bus)
        executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

        # Simulate voice command: "status"
        task = planner.plan_command("status")
        assert task is not None
        result = executor.execute_task(task)
        assert result.success is True

        # Speech should have spoken a status report
        assert len(speech.spoken) >= 1
        report_text = speech.spoken[-1][0]
        assert "SIMON" in report_text

    def test_navigate_command_flow(self, event_bus, state_machine, capabilities):
        speech = MockSpeechManager()
        nav = MockNavPipeline()

        registry = ActionRegistry()
        register_builtin_handlers(
            registry=registry,
            speech_manager=speech,
            nav_pipeline=nav,
            capabilities=capabilities,
            state_machine=state_machine,
        )

        planner = TaskPlanner(event_bus=event_bus)
        executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

        # Simulate voice command: "navigate to library"
        task = planner.plan_command("navigate", {"destination": "library"})
        assert task is not None
        assert len(task.subtasks) == 2

        result = executor.execute_task(task)
        assert result.success is True
        assert nav.destination == "library"
        assert nav.navigating is True

        # First subtask should have spoken "Finding route to library"
        assert any("library" in text.lower() for text, _ in speech.spoken)

    def test_stop_command_flow(self, event_bus, capabilities):
        sm = StateMachine(initial_state=SystemState.STARTING)
        sm.transition(SystemState.INITIALIZING)
        sm.transition(SystemState.READY)

        speech = MockSpeechManager()
        registry = ActionRegistry()
        register_builtin_handlers(
            registry=registry,
            speech_manager=speech,
            state_machine=sm,
        )

        planner = TaskPlanner(event_bus=event_bus)
        executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

        task = planner.plan_command("stop")
        assert task is not None
        result = executor.execute_task(task)
        assert result.success is True

        # Should have spoken "Goodbye."
        assert any("goodbye" in text.lower() for text, _ in speech.spoken)
        # State should transition to SHUTTING_DOWN
        assert sm.state == SystemState.SHUTTING_DOWN

    def test_cancel_nav_command_flow(self, event_bus, state_machine, capabilities):
        speech = MockSpeechManager()
        nav = MockNavPipeline()
        nav.navigating = True

        registry = ActionRegistry()
        register_builtin_handlers(
            registry=registry,
            speech_manager=speech,
            nav_pipeline=nav,
        )

        planner = TaskPlanner(event_bus=event_bus)
        executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

        task = planner.plan_command("cancel_nav")
        assert task is not None
        result = executor.execute_task(task)
        assert result.success is True
        assert nav.navigating is False

    def test_unknown_command_returns_none(self, event_bus):
        planner = TaskPlanner(event_bus=event_bus)
        task = planner.plan_command("teleport")
        assert task is None

    def test_read_text_without_ocr(self, event_bus, state_machine):
        """read_text should fail gracefully when OCR unavailable."""
        capabilities = CapabilityRegistry()
        capabilities.register("capability.ocr", available=False)

        speech = MockSpeechManager()
        registry = ActionRegistry()
        register_builtin_handlers(
            registry=registry,
            speech_manager=speech,
            capabilities=capabilities,
        )

        planner = TaskPlanner(event_bus=event_bus)
        executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

        task = planner.plan_command("read_text")
        assert task is not None
        result = executor.execute_task(task)
        assert result.success is False
        assert any("not available" in text.lower() for text, _ in speech.spoken)

    def test_state_machine_lifecycle(self):
        """Verify the full state lifecycle works."""
        sm = StateMachine(initial_state=SystemState.STARTING)
        sm.transition(SystemState.INITIALIZING)
        sm.transition(SystemState.READY)
        assert sm.state == SystemState.READY

        sm.transition(SystemState.SHUTTING_DOWN)
        assert sm.state == SystemState.SHUTTING_DOWN

        sm.transition(SystemState.STOPPED)
        assert sm.state == SystemState.STOPPED

    def test_degraded_state_when_capabilities_missing(self):
        """State should go to DEGRADED when some capabilities are unavailable."""
        sm = StateMachine(initial_state=SystemState.STARTING)
        sm.transition(SystemState.INITIALIZING)

        capabilities = CapabilityRegistry()
        capabilities.register("capability.camera", available=False, reason="no device")
        capabilities.register("capability.gps", available=True)

        unavailable = capabilities.list_unavailable()
        assert len(unavailable) >= 1

        sm.transition(SystemState.DEGRADED)
        assert sm.state == SystemState.DEGRADED

    def test_watchdog_integration(self, event_bus):
        """Watchdog should detect stale threads."""
        dead_events = []
        event_bus.subscribe(
            "health.thread_dead",
            lambda e: dead_events.append(e),
            source="test",
        )

        watchdog = Watchdog(event_bus=event_bus, interval_s=0.1, timeout_s=0.2)
        watchdog.register_thread("dying_thread")
        watchdog.start()

        time.sleep(0.5)
        watchdog.stop()

        assert len(dead_events) >= 1

    def test_model_registry_tracking(self):
        """ModelRegistry tracks loaded models."""
        reg = ModelRegistry()
        reg.register("yolo", "cuda", 100.0, 1)
        reg.register("ocr", "cpu", 50.0, 5)

        assert reg.model_count == 2
        assert reg.get_total_memory_mb("cuda") == 100.0
        assert reg.get_total_memory_mb("cpu") == 50.0

        candidate = reg.get_eviction_candidate()
        assert candidate is not None
        assert candidate.name == "ocr"  # Lower priority = more likely evicted
