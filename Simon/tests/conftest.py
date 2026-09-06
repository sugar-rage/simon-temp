"""System-wide test fixtures.

Provides mock subsystems for integration testing without
requiring hardware (camera, GPS, microphone) or ML models.
"""

from __future__ import annotations

import pytest

from core.config.system_config import SystemConfig
from core.events.event_bus import EventBus
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.resources.model_registry import ModelRegistry
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor
from core.actions.action_registry import ActionRegistry
from core.planner.task_planner import TaskPlanner
from core.actions.action_executor import ActionExecutor
from core.plugins.hooks import HookRegistry
from core.metrics.collector import SystemMetricsCollector


class MockSpeechManager:
    """Mock SpeechManager that records speak/command calls."""

    def __init__(self):
        self.spoken = []
        self.commands = []
        self._started = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def speak(self, text, priority=6):
        self.spoken.append((text, priority))

    def get_command(self):
        if self.commands:
            return self.commands.pop(0)
        return None

    def queue_command(self, action, args=""):
        self.commands.append({"action": action, "args": args})


class MockNavigationPipeline:
    """Mock navigation pipeline."""

    def __init__(self):
        self.navigating = False
        self.destination = None

    def start(self):
        pass

    def stop(self):
        self.navigating = False

    def navigate_to(self, destination):
        self.destination = destination
        self.navigating = True

    def cancel(self):
        self.navigating = False
        self.destination = None

    @property
    def is_navigating(self):
        return self.navigating


@pytest.fixture
def system_config():
    return SystemConfig()


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def state_machine():
    return StateMachine(initial_state=SystemState.STARTING)


@pytest.fixture
def capabilities():
    return CapabilityRegistry()


@pytest.fixture
def model_registry():
    return ModelRegistry()


@pytest.fixture
def mock_speech():
    return MockSpeechManager()


@pytest.fixture
def mock_nav():
    return MockNavigationPipeline()


@pytest.fixture(autouse=True)
def reset_metrics():
    yield
    SystemMetricsCollector.reset_instance()
