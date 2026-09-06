"""Unit tests for HealthMonitor — status reports and diagnostics."""

from __future__ import annotations

import pytest

from core.events.event_bus import EventBus
from core.capabilities.registry import CapabilityRegistry
from core.state.state_machine import StateMachine, SystemState
from core.resources.model_registry import ModelRegistry
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def capabilities():
    return CapabilityRegistry()


@pytest.fixture
def state_machine():
    sm = StateMachine(initial_state=SystemState.STARTING)
    sm.transition(SystemState.INITIALIZING)
    sm.transition(SystemState.READY)
    return sm


@pytest.fixture
def watchdog(event_bus):
    wd = Watchdog(event_bus=event_bus, interval_s=60.0, timeout_s=120.0)
    yield wd
    if wd.is_running:
        wd.stop()


@pytest.fixture
def model_registry():
    return ModelRegistry()


@pytest.fixture
def monitor(capabilities, watchdog, model_registry, state_machine, event_bus):
    return HealthMonitor(
        capabilities=capabilities,
        watchdog=watchdog,
        model_registry=model_registry,
        state_machine=state_machine,
        event_bus=event_bus,
    )


class TestHealthMonitor:
    def test_status_report_ready_state(self, monitor):
        report = monitor.get_status_report()
        assert "SIMON is running" in report

    def test_status_report_camera_available(self, monitor, capabilities):
        capabilities.register("capability.camera", available=True)
        report = monitor.get_status_report()
        assert "Camera active" in report

    def test_status_report_camera_unavailable(self, monitor, capabilities):
        capabilities.register("capability.camera", available=False)
        report = monitor.get_status_report()
        assert "Camera unavailable" in report

    def test_status_report_gps_available(self, monitor, capabilities):
        capabilities.register("capability.gps", available=True)
        report = monitor.get_status_report()
        assert "GPS active" in report

    def test_status_report_gps_unavailable(self, monitor, capabilities):
        capabilities.register("capability.gps", available=False, reason="no device")
        report = monitor.get_status_report()
        assert "GPS unavailable" in report

    def test_status_report_all_systems(self, monitor, capabilities):
        capabilities.register("capability.camera", available=True)
        capabilities.register("capability.detection", available=True)
        capabilities.register("capability.gps", available=True)
        capabilities.register("capability.ocr", available=True)
        capabilities.register("capability.face_recognition", available=True)
        report = monitor.get_status_report()
        assert "All systems operational" in report

    def test_status_report_degraded_state(self, capabilities, watchdog, model_registry, event_bus):
        sm = StateMachine(initial_state=SystemState.STARTING)
        sm.transition(SystemState.INITIALIZING)
        sm.transition(SystemState.DEGRADED)
        monitor = HealthMonitor(
            capabilities=capabilities,
            watchdog=watchdog,
            model_registry=model_registry,
            state_machine=sm,
            event_bus=event_bus,
        )
        report = monitor.get_status_report()
        assert "degraded" in report.lower()

    def test_detailed_report_structure(self, monitor, capabilities):
        capabilities.register("capability.camera", available=True)
        report = monitor.get_detailed_report()
        assert "state" in report
        assert report["state"] == "READY"
        assert "uptime_s" in report
        assert report["uptime_s"] >= 0
        assert "capabilities" in report
        assert "threads" in report
        assert "models" in report
        assert "model_count" in report
        assert "watchdog_running" in report

    def test_detailed_report_models(self, monitor, model_registry):
        model_registry.register("yolo", "cuda", 150.0, 1)
        report = monitor.get_detailed_report()
        assert report["model_count"] == 1
        assert report["models"][0]["name"] == "yolo"

    def test_check_health_publishes_event(self, monitor, event_bus):
        events = []
        event_bus.subscribe(
            "health.check", lambda e: events.append(e), source="test",
        )
        event_bus.start()
        monitor.check_health()
        import time
        time.sleep(0.2)
        event_bus.stop()
        assert len(events) >= 1
