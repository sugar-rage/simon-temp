"""Speech roundtrip benchmarks.

Measures the command processing pipeline latency:
- Voice command parsing → TaskPlanner decomposition
- TaskPlanner → ActionExecutor dispatch
- Full command roundtrip (parse → plan → execute → response)

Target: Voice command → Parse → Event → Action → Response < 500ms

Usage::

    python -m pytest tests/performance/bench_speech_roundtrip.py -v -s
"""

from __future__ import annotations

import time
import statistics
import pytest

from core.events.event_bus import EventBus
from core.models.actions import Action, ActionResult
from core.actions.action_registry import ActionRegistry
from core.actions.action_executor import ActionExecutor
from core.actions.action_handlers import register_builtin_handlers
from core.planner.task_planner import TaskPlanner
from core.planner.task import Task, TaskType
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.resources.model_registry import ModelRegistry
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor


# ── Fixtures ─────────────────────────────────────────────────────────


class MockSpeech:
    """Minimal mock for benchmarking (no I/O)."""
    def speak(self, text, priority=6):
        pass  # No-op for speed benchmarking

    def start(self):
        pass

    def stop(self):
        pass


class MockNav:
    """Minimal mock for benchmarking."""
    navigating = False
    destination = None

    def navigate_to(self, dest):
        self.destination = dest
        self.navigating = True

    def cancel(self):
        self.navigating = False


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def wired_system(event_bus):
    """Return a fully wired planner + executor with mock subsystems."""
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

    speech = MockSpeech()
    nav = MockNav()

    model_reg = ModelRegistry()
    watchdog = Watchdog(event_bus=event_bus, interval_s=60, timeout_s=120)
    health = HealthMonitor(
        capabilities=caps,
        watchdog=watchdog,
        model_registry=model_reg,
        state_machine=sm,
        event_bus=event_bus,
    )

    registry = ActionRegistry()
    register_builtin_handlers(
        registry=registry,
        speech_manager=speech,
        nav_pipeline=nav,
        capabilities=caps,
        health_monitor=health,
        state_machine=sm,
    )

    planner = TaskPlanner(event_bus=event_bus)
    executor = ActionExecutor(action_registry=registry, event_bus=event_bus)

    return planner, executor, speech, nav


# ── Benchmarks ───────────────────────────────────────────────────────


class TestSpeechRoundtripBenchmarks:
    """Benchmark the voice command processing pipeline."""

    ITERATIONS = 200

    def test_planner_decomposition_latency(self, wired_system):
        """Measure TaskPlanner.plan_command() latency.

        Target: < 1ms per planning call.
        """
        planner, _, _, _ = wired_system
        latencies = []

        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            task = planner.plan_command("status")
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n  Planner Decomposition ({self.ITERATIONS} iterations):")
        print(f"    avg: {avg:.3f}ms  p95: {p95:.3f}ms")

        assert avg < 5.0, f"Planning too slow: {avg:.3f}ms"

    def test_executor_dispatch_latency(self, wired_system):
        """Measure ActionExecutor.execute() latency for a single action.

        Target: < 5ms per action dispatch (excludes actual handler work).
        """
        _, executor, _, _ = wired_system
        latencies = []

        for _ in range(self.ITERATIONS):
            action = Action(action_type="speak", args={"text": "test"})
            t0 = time.perf_counter()
            result = executor.execute(action)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n  Executor Dispatch ({self.ITERATIONS} iterations):")
        print(f"    avg: {avg:.3f}ms  p95: {p95:.3f}ms")

        assert avg < 10.0, f"Execution too slow: {avg:.3f}ms"

    def test_full_status_roundtrip(self, wired_system):
        """Measure full roundtrip: plan → execute for 'status' command.

        Target: < 10ms (excludes real TTS latency).
        """
        planner, executor, _, _ = wired_system
        latencies = []

        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()

            task = planner.plan_command("status")
            assert task is not None
            result = executor.execute_task(task)

            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\n  Full Status Roundtrip ({self.ITERATIONS} iterations):")
        print(f"    avg: {avg:.2f}ms  p95: {p95:.2f}ms  p99: {p99:.2f}ms")

        assert avg < 50.0, f"Status roundtrip too slow: {avg:.2f}ms"

    def test_full_navigate_roundtrip(self, wired_system):
        """Measure full roundtrip for 'navigate' (2 subtasks)."""
        planner, executor, _, nav = wired_system
        latencies = []

        for i in range(self.ITERATIONS):
            nav.navigating = False
            t0 = time.perf_counter()

            task = planner.plan_command("navigate", {"destination": "library"})
            assert task is not None
            result = executor.execute_task(task)

            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n  Full Navigate Roundtrip ({self.ITERATIONS} iterations):")
        print(f"    avg: {avg:.2f}ms  p95: {p95:.2f}ms")

        assert avg < 50.0, f"Navigate roundtrip too slow: {avg:.2f}ms"

    def test_command_throughput(self, wired_system):
        """Measure max commands/sec throughput."""
        planner, executor, _, _ = wired_system

        t0 = time.perf_counter()
        commands = 0
        while (time.perf_counter() - t0) < 2.0:
            task = planner.plan_command("status")
            if task:
                executor.execute_task(task)
            commands += 1

        elapsed = time.perf_counter() - t0
        cps = commands / elapsed

        print(f"\n  Command Throughput: {cps:.0f} commands/sec ({commands} in {elapsed:.2f}s)")
        assert cps > 50, f"Command throughput {cps:.0f}/sec is below 50/sec"

    def test_planner_concurrent_tasks(self, wired_system):
        """Measure planner performance with many active tasks."""
        planner, executor, _, _ = wired_system

        # Pre-fill with active tasks
        for _ in range(50):
            planner.plan_command("status")

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            task = planner.plan_command("status")
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        print(f"\n  Planner with 50+ active tasks: avg {avg:.3f}ms")
        assert avg < 10.0, f"Planner slows under load: {avg:.3f}ms"
