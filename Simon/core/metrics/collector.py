"""System-wide metrics collector.

Provides a lightweight, thread-safe metrics system for SIMON.
Mirrors the ``speech/monitoring/metrics.py`` pattern but generalized
for all subsystems.

Namespaces::

    vision.frame_count          — total frames processed
    vision.detection_latency_ms — per-frame detection time
    vision.face_count           — faces recognized
    navigation.gps_fix_count    — GPS fixes received
    navigation.route_compute_ms — route computation time
    core.event_count            — events dispatched
    core.event_latency_ms       — event dispatch latency
    speech.*                    — (collected by speech subsystem)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional


class SystemMetricsCollector:
    """Thread-safe metrics collector with counter, gauge, and timer support.

    This is the only singleton-like object in SIMON.  Use
    ``SystemMetricsCollector.get()`` to access the shared instance,
    or construct directly for testing.

    Thread-safety: all mutations are protected by a lock.
    """

    _instance: Optional[SystemMetricsCollector] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> SystemMetricsCollector:
        """Return the shared metrics collector instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, list[float]] = defaultdict(list)
        self._timer_max_samples = 1000

    # ── Counters (monotonically increasing) ──────────────────────────

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        """Return the current value of a counter."""
        with self._lock:
            return self._counters.get(name, 0)

    # ── Gauges (point-in-time values) ────────────────────────────────

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value."""
        with self._lock:
            self._gauges[name] = value

    def get_gauge(self, name: str) -> float:
        """Return the current value of a gauge."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    # ── Timers (latency tracking) ────────────────────────────────────

    def record_time(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        with self._lock:
            samples = self._timers[name]
            samples.append(duration_ms)
            # Prevent unbounded growth
            if len(samples) > self._timer_max_samples:
                self._timers[name] = samples[-self._timer_max_samples:]

    def timer(self, name: str) -> _TimerContext:
        """Context manager to measure elapsed time.

        Usage::

            with metrics.timer("vision.detection_latency_ms"):
                detections = model.detect(frame)
        """
        return _TimerContext(self, name)

    def get_timer_stats(self, name: str) -> dict[str, float]:
        """Return min/max/avg/count for a timer."""
        with self._lock:
            samples = self._timers.get(name, [])
            if not samples:
                return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
            return {
                "count": len(samples),
                "min": min(samples),
                "max": max(samples),
                "avg": sum(samples) / len(samples),
            }

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict]:
        """Return a complete metrics snapshot (for diagnostics/health)."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {
                    name: {
                        "count": len(samples),
                        "avg": (
                            sum(samples) / len(samples) if samples else 0.0
                        ),
                    }
                    for name, samples in self._timers.items()
                },
            }

    def reset(self) -> None:
        """Clear all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()


class _TimerContext:
    """Context manager for timing code blocks."""

    __slots__ = ("_collector", "_name", "_start")

    def __init__(self, collector: SystemMetricsCollector, name: str) -> None:
        self._collector = collector
        self._name = name
        self._start = 0.0

    def __enter__(self) -> _TimerContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._collector.record_time(self._name, elapsed_ms)
