"""Memory usage profiling.

Tracks object counts and memory footprint of core data structures:
- ModelRegistry under load
- EventBus with many subscribers
- TaskPlanner with many active tasks
- DetectionTracker with object churn

Usage::

    python -m pytest tests/performance/bench_memory_usage.py -v -s
"""

from __future__ import annotations

import sys
import time
import pytest

from core.events.event_bus import EventBus
from core.models.detection import Detection
from core.models.events import Event
from core.models.enums import Priority
from core.resources.model_registry import ModelRegistry
from core.planner.task_planner import TaskPlanner
from vision.detection.tracker import DetectionTracker


# ── Helpers ──────────────────────────────────────────────────────────


def _object_size_bytes(obj) -> int:
    """Approximate deep size of an object in bytes."""
    try:
        return sys.getsizeof(obj)
    except TypeError:
        return 0


# ── Memory Benchmarks ────────────────────────────────────────────────


class TestMemoryBenchmarks:
    """Verify that data structures don't grow unboundedly."""

    def test_model_registry_memory(self):
        """ModelRegistry should have bounded memory per model."""
        registry = ModelRegistry()

        for i in range(100):
            registry.register(f"model_{i}", "cuda", 100.0, i % 10)

        assert registry.model_count == 100
        total_mb = registry.get_total_memory_mb()
        print(f"\n  ModelRegistry: {registry.model_count} models, {total_mb:.0f} MB tracked")

        # Unregister half
        for i in range(50):
            registry.unregister(f"model_{i}")

        assert registry.model_count == 50
        print(f"  After eviction: {registry.model_count} models, {registry.get_total_memory_mb():.0f} MB")

    def test_event_bus_subscriber_memory(self):
        """EventBus should not leak memory with many subscribers."""
        bus = EventBus(rate_limit_s=0.0, enable_metrics=False)

        handlers = []
        for i in range(200):
            handler = lambda e, idx=i: None
            handlers.append(handler)
            bus.subscribe(f"topic.{i % 20}", handler, source=f"bench_{i}")

        # Unsubscribe all
        for handler in handlers:
            bus.unsubscribe(handler)

        print(f"\n  EventBus: 200 subscriptions added and removed")

    def test_task_planner_bounded_tasks(self):
        """TaskPlanner should prune completed tasks and not grow unboundedly."""
        bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
        planner = TaskPlanner(event_bus=bus, max_active_tasks=50)

        # Create and complete many tasks
        for i in range(200):
            task = planner.plan_command("status")
            if task:
                planner.on_task_complete(task.task_id, {"i": i})

        active = planner.get_active_tasks()
        print(f"\n  TaskPlanner: 200 tasks created/completed, {len(active)} active remaining")

        # Active tasks should be bounded
        assert len(active) <= 50, (
            f"TaskPlanner has {len(active)} active tasks, expected ≤ 50"
        )

    def test_detection_tracker_object_churn(self):
        """DetectionTracker should age out old tracks."""
        tracker = DetectionTracker()
        classes = ["person", "car", "dog", "bicycle", "chair"]

        # Simulate 500 frames with changing detections
        for frame in range(500):
            # Every 50 frames, introduce new objects at different positions
            offset = (frame // 50) * 200
            detections = [
                Detection(
                    cls=classes[j % len(classes)],
                    confidence=0.8,
                    bbox=(offset + j * 50, 100, offset + j * 50 + 40, 180),
                )
                for j in range(5)
            ]
            tracked = tracker.update(detections)

        active = tracker.active_count
        print(f"\n  DetectionTracker: 500 frames processed, {active} active tracks")

        # Should not have accumulated all historical tracks
        assert active <= 50, (
            f"Tracker has {active} active tracks after 500 frames — possible leak"
        )

    def test_metrics_collector_bounded(self):
        """SystemMetricsCollector timers should be bounded."""
        from core.metrics.collector import SystemMetricsCollector

        metrics = SystemMetricsCollector()

        # Record 5000 timer samples
        for i in range(5000):
            metrics.record_time("test.bench", float(i % 100))

        stats = metrics.get_timer_stats("test.bench")
        print(f"\n  MetricsCollector: 5000 samples recorded, {stats['count']} retained")

        # Timer samples should be bounded (max 1000 by default)
        assert stats["count"] <= 1000, (
            f"Timer has {stats['count']} samples, expected ≤ 1000"
        )

    def test_event_creation_memory(self):
        """Event objects should have consistent, bounded size."""
        events = []
        for i in range(1000):
            events.append(Event(
                topic=f"bench.topic.{i % 10}",
                priority=Priority.INFORMATIONAL,
                source="bench",
                data={"value": i, "name": f"item_{i}"},
            ))

        sizes = [_object_size_bytes(e) for e in events]
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        print(f"\n  Event objects: avg {avg_size:.0f} bytes each, {len(events)} created")
        assert avg_size < 1024, f"Events are too large: {avg_size:.0f} bytes avg"
