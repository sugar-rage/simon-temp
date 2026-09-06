"""Shared test fixtures for core infrastructure tests."""

from __future__ import annotations

import pytest

from core.events.event_bus import EventBus
from core.models.events import Event
from core.models.enums import Priority
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.metrics.collector import SystemMetricsCollector


@pytest.fixture
def event_bus():
    """Create a fresh EventBus for testing (no background thread)."""
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def running_event_bus():
    """Create an EventBus with its dispatch thread started."""
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    bus.start()
    yield bus
    bus.stop()
    bus.clear()


@pytest.fixture
def state_machine():
    """Create a fresh StateMachine in STARTING state."""
    return StateMachine(initial_state=SystemState.STARTING)


@pytest.fixture
def ready_state_machine():
    """Create a StateMachine already transitioned to READY."""
    sm = StateMachine(initial_state=SystemState.STARTING)
    sm.transition(SystemState.INITIALIZING)
    sm.transition(SystemState.READY)
    return sm


@pytest.fixture
def capability_registry():
    """Create a fresh CapabilityRegistry."""
    return CapabilityRegistry()


@pytest.fixture
def metrics():
    """Create a fresh metrics collector (not the singleton)."""
    collector = SystemMetricsCollector()
    yield collector
    collector.reset()


@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset the metrics singleton between tests."""
    yield
    SystemMetricsCollector.reset_instance()


def make_event(
    topic: str = "test.event",
    priority: int = Priority.INFORMATIONAL,
    **data,
) -> Event:
    """Helper to create test events quickly."""
    return Event(topic=topic, priority=priority, data=data, source="test")
