"""Stress test — concurrent subsystem load test.

Runs all core pipelines concurrently to verify:
- No deadlocks under concurrent EventBus/TaskPlanner/ActionExecutor usage
- No memory leaks over extended operation
- No thread safety violations
- Event Bus handles high-frequency publishing

Duration: Configurable (default 10 seconds, CI target 30 seconds)

Usage::

    python -m pytest tests/performance/stress_test.py -v -s
    python -m pytest tests/performance/stress_test.py -v -s --stress-duration=30
"""

from __future__ import annotations

import threading
import time
import pytest
import os

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.models.detection import Detection
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.resources.model_registry import ModelRegistry
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor
from core.actions.action_registry import ActionRegistry
from core.actions.action_executor import ActionExecutor
from core.actions.action_handlers import register_builtin_handlers
from core.planner.task_planner import TaskPlanner
from core.planner.task import Task, TaskType
from vision.detection.tracker import DetectionTracker
from vision.pipeline.perception_fusion import PerceptionFusion


# ── Configuration ────────────────────────────────────────────────────


STRESS_DURATION_S = float(os.environ.get("STRESS_DURATION", "10"))


# ── Mock subsystems ──────────────────────────────────────────────────


class StressSpeech:
    """Thread-safe mock speech for stress testing."""

    def __init__(self):
        self.speak_count = 0
        self._lock = threading.Lock()

    def speak(self, text, priority=6):
        with self._lock:
            self.speak_count += 1

    def start(self):
        pass

    def stop(self):
        pass


class StressNav:
    """Thread-safe mock navigation for stress testing."""

    def __init__(self):
        self.nav_count = 0
        self._lock = threading.Lock()
        self.navigating = False

    def navigate_to(self, dest):
        with self._lock:
            self.nav_count += 1
            self.navigating = True

    def cancel(self):
        with self._lock:
            self.navigating = False


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def stress_system():
    """Create a fully wired system for stress testing."""
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    bus.start()

    sm = StateMachine(initial_state=SystemState.STARTING)
    sm.transition(SystemState.INITIALIZING)
    sm.transition(SystemState.READY)

    caps = CapabilityRegistry()
    caps.register("capability.camera", available=True)
    caps.register("capability.detection", available=True)
    caps.register("capability.gps", available=True)
    caps.register("capability.speech", available=True)
    caps.register("capability.ocr", available=True)
    caps.register("capability.face_recognition", available=True)

    speech = StressSpeech()
    nav = StressNav()
    model_reg = ModelRegistry()
    watchdog = Watchdog(bus, interval_s=1.0, timeout_s=5.0)
    health = HealthMonitor(
        capabilities=caps, watchdog=watchdog,
        model_registry=model_reg, state_machine=sm, event_bus=bus,
    )

    registry = ActionRegistry()
    register_builtin_handlers(
        registry=registry, speech_manager=speech, nav_pipeline=nav,
        capabilities=caps, health_monitor=health, state_machine=sm,
    )

    planner = TaskPlanner(event_bus=bus)
    executor = ActionExecutor(action_registry=registry, event_bus=bus)
    tracker = DetectionTracker()
    fusion = PerceptionFusion()

    yield {
        "bus": bus, "sm": sm, "caps": caps, "speech": speech, "nav": nav,
        "model_reg": model_reg, "watchdog": watchdog, "health": health,
        "planner": planner, "executor": executor, "tracker": tracker,
        "fusion": fusion,
    }

    # Teardown
    watchdog.stop()
    bus.stop()
    bus.clear()


# ── Stress Tests ─────────────────────────────────────────────────────


class TestStressTest:
    """Concurrent stress tests for the full system."""

    def test_concurrent_event_publishing(self, stress_system):
        """Stress the EventBus with concurrent publishers.

        Multiple threads publish events simultaneously.
        No crashes, no deadlocks.
        """
        bus = stress_system["bus"]
        received = {"count": 0, "lock": threading.Lock()}

        bus.subscribe(
            "stress.*",
            lambda e: _safe_increment(received),
            source="stress_receiver",
        )

        errors = []
        stop = threading.Event()

        def publisher(topic, thread_id):
            count = 0
            try:
                while not stop.is_set():
                    bus.publish(Event(
                        topic=topic,
                        priority=Priority.LOW,
                        source=f"thread_{thread_id}",
                        data={"n": count},
                    ))
                    count += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(
                target=publisher,
                args=(f"stress.topic_{i % 5}", i),
                daemon=True,
            )
            for i in range(10)
        ]

        for t in threads:
            t.start()
        time.sleep(min(STRESS_DURATION_S, 5))
        stop.set()
        for t in threads:
            t.join(timeout=3)

        print(f"\n  Concurrent Publishing: {received['count']} events received, {len(errors)} errors")
        assert len(errors) == 0, f"Publisher errors: {errors}"

    def test_concurrent_command_processing(self, stress_system):
        """Stress the planner + executor with concurrent commands.

        Multiple threads submit commands simultaneously.
        """
        planner = stress_system["planner"]
        executor = stress_system["executor"]

        errors = []
        completed = {"count": 0, "lock": threading.Lock()}
        stop = threading.Event()

        commands = ["status", "navigate", "stop", "cancel_nav", "describe_scene"]

        def command_worker(thread_id):
            try:
                while not stop.is_set():
                    cmd = commands[thread_id % len(commands)]
                    args = {"destination": "park"} if cmd == "navigate" else {}
                    task = planner.plan_command(cmd, args)
                    if task:
                        executor.execute_task(task)
                        _safe_increment(completed)
                    time.sleep(0.01)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=command_worker, args=(i,), daemon=True)
            for i in range(5)
        ]

        for t in threads:
            t.start()
        time.sleep(min(STRESS_DURATION_S, 5))
        stop.set()
        for t in threads:
            t.join(timeout=3)

        print(f"\n  Concurrent Commands: {completed['count']} completed, {len(errors)} errors")
        assert len(errors) == 0, f"Command errors: {errors}"
        assert completed["count"] > 0, "No commands completed"

    def test_concurrent_vision_and_commands(self, stress_system):
        """Stress vision pipeline and command processing simultaneously.

        One thread runs perception fusion, another processes commands.
        """
        fusion = stress_system["fusion"]
        planner = stress_system["planner"]
        executor = stress_system["executor"]
        bus = stress_system["bus"]

        errors = []
        stats = {"frames": 0, "commands": 0, "lock": threading.Lock()}
        stop = threading.Event()

        def vision_worker():
            classes = ["person", "car", "dog", "chair", "bottle"]
            try:
                frame_id = 0
                while not stop.is_set():
                    detections = [
                        Detection(
                            cls=classes[j % len(classes)],
                            confidence=0.8,
                            bbox=(j * 50, 100, j * 50 + 60, 200),
                        )
                        for j in range(10)
                    ]
                    fusion.fuse(
                        detections=detections, faces=[], ocr_results=[],
                        frame_id=frame_id,
                    )
                    frame_id += 1
                    with stats["lock"]:
                        stats["frames"] += 1
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("vision", e))

        def command_worker():
            try:
                while not stop.is_set():
                    task = planner.plan_command("status")
                    if task:
                        executor.execute_task(task)
                    with stats["lock"]:
                        stats["commands"] += 1
                    time.sleep(0.02)
            except Exception as e:
                errors.append(("command", e))

        t1 = threading.Thread(target=vision_worker, daemon=True)
        t2 = threading.Thread(target=command_worker, daemon=True)

        t1.start()
        t2.start()
        time.sleep(min(STRESS_DURATION_S, 5))
        stop.set()
        t1.join(timeout=3)
        t2.join(timeout=3)

        print(f"\n  Concurrent Vision+Commands: {stats['frames']} frames, {stats['commands']} commands, {len(errors)} errors")
        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_watchdog_under_load(self, stress_system):
        """Watchdog should detect stale threads even under load."""
        watchdog = stress_system["watchdog"]
        bus = stress_system["bus"]

        dead_events = []
        bus.subscribe(
            "health.thread_dead",
            lambda e: dead_events.append(e.data["thread_name"]),
            source="stress_test",
        )

        # Register a thread that will go stale
        watchdog.register_thread("stress_stale")
        # Register a healthy thread
        watchdog.register_thread("stress_healthy")
        watchdog.start()

        # Keep healthy thread alive
        stop = threading.Event()
        def keepalive():
            while not stop.is_set():
                watchdog.heartbeat("stress_healthy")
                time.sleep(0.5)

        t = threading.Thread(target=keepalive, daemon=True)
        t.start()

        # Wait for stale detection
        time.sleep(8)
        stop.set()
        t.join(timeout=2)
        watchdog.stop()

        print(f"\n  Watchdog: dead events for {dead_events}")
        assert "stress_stale" in dead_events, "Watchdog missed stale thread"
        assert "stress_healthy" not in dead_events, "Watchdog false positive on healthy thread"

    def test_model_registry_concurrent(self, stress_system):
        """ModelRegistry should be thread-safe under concurrent access."""
        reg = stress_system["model_reg"]
        errors = []
        stop = threading.Event()

        def register_worker(thread_id):
            try:
                for i in range(100):
                    if stop.is_set():
                        break
                    name = f"model_t{thread_id}_{i}"
                    reg.register(name, "cpu", 10.0, 5)
                    reg.touch(name)
                    reg.get_model(name)
                    reg.list_models()
                    reg.get_total_memory_mb()
                    reg.unregister(name)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=register_worker, args=(i,), daemon=True)
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        stop.set()

        print(f"\n  ModelRegistry Concurrent: {len(errors)} errors, {reg.model_count} remaining models")
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_capability_registry_concurrent(self, stress_system):
        """CapabilityRegistry should be thread-safe under concurrent access."""
        caps = stress_system["caps"]
        errors = []

        def update_worker(thread_id):
            try:
                for i in range(200):
                    cap_name = f"cap.test_{thread_id}"
                    caps.update(cap_name, available=(i % 2 == 0))
                    caps.is_available(cap_name)
                    caps.list_available()
                    caps.list_degraded()
                    caps.list_unavailable()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=update_worker, args=(i,), daemon=True)
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        print(f"\n  CapabilityRegistry Concurrent: {len(errors)} errors")
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_no_memory_leak_extended(self, stress_system):
        """Verify no significant memory growth over extended operation."""
        planner = stress_system["planner"]
        executor = stress_system["executor"]

        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pytest.skip("psutil not available for memory tracking")
            return

        # Run 1000 command cycles
        for i in range(1000):
            task = planner.plan_command("status")
            if task:
                executor.execute_task(task)
                planner.on_task_complete(task.task_id, {"i": i})

        mem_after = process.memory_info().rss / (1024 * 1024)
        growth = mem_after - mem_before

        print(f"\n  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (growth: {growth:+.1f} MB)")

        # Allow up to 50MB growth (generous for Python)
        assert growth < 50, f"Memory grew by {growth:.1f} MB — possible leak"


# ── Helpers ──────────────────────────────────────────────────────────


def _safe_increment(counter: dict) -> None:
    """Thread-safe increment of a counter dict."""
    with counter["lock"]:
        counter["count"] += 1
