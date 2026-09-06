"""
Metrics collection for the SIMON speech subsystem.

Provides lightweight, thread-safe counters, gauges, and histograms for
monitoring pipeline performance in real time.  Metrics are stored in memory
and can be queried by the dashboard, health monitor, or exported to logs.

Design decision: We implement a simple custom metrics layer rather than
depending on Prometheus client or similar, because:
1. Edge deployment — no metrics server to push to.
2. Minimal overhead — this runs on a laptop alongside heavy ML models.
3. Thread-safe without external dependencies.

Each metric is identified by a string name and is globally accessible via
``get_collector()``.  Components record metrics like::

    metrics = get_collector()
    metrics.histogram("stt.latency_ms", elapsed_ms)
    metrics.counter("stt.redecode_count")
    metrics.gauge("stt.confidence", 0.92)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HistogramBucket:
    """Running statistics for a histogram metric.

    Maintains a sliding window of recent values for percentile calculation,
    plus cumulative sum/count for overall averages.
    """

    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_sum: float = 0.0
    total_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, value: float) -> None:
        with self._lock:
            self.values.append(value)
            self.total_sum += value
            self.total_count += 1

    @property
    def mean(self) -> float:
        with self._lock:
            if self.total_count == 0:
                return 0.0
            return self.total_sum / self.total_count

    @property
    def recent_mean(self) -> float:
        """Mean of the recent window (last 1000 values)."""
        with self._lock:
            if not self.values:
                return 0.0
            return sum(self.values) / len(self.values)

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile of the recent window.

        Args:
            p: Percentile (0-100).
        """
        with self._lock:
            if not self.values:
                return 0.0
            sorted_vals = sorted(self.values)
            idx = int(len(sorted_vals) * p / 100.0)
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def min_val(self) -> float:
        with self._lock:
            return min(self.values) if self.values else 0.0

    @property
    def max_val(self) -> float:
        with self._lock:
            return max(self.values) if self.values else 0.0


class MetricsCollector:
    """Central, thread-safe metrics collection point.

    Supports three metric types:

    - **Counter**: Monotonically increasing integer (e.g. total detections).
    - **Gauge**: Current value that can go up or down (e.g. CPU usage).
    - **Histogram**: Distribution of values (e.g. latency).
    - **Label**: String-valued label (e.g. current STT engine name).

    All operations are thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._labels: dict[str, str] = {}
        self._histograms: dict[str, HistogramBucket] = {}
        self._timestamps: dict[str, float] = {}  # Last update time per metric

    # ------------------------------------------------------------------ #
    #  Counter
    # ------------------------------------------------------------------ #

    def counter(self, name: str, increment: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += increment
            self._timestamps[name] = time.monotonic()

    def get_counter(self, name: str) -> int:
        """Read a counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    # ------------------------------------------------------------------ #
    #  Gauge
    # ------------------------------------------------------------------ #

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to an absolute value."""
        with self._lock:
            self._gauges[name] = value
            self._timestamps[name] = time.monotonic()

    def get_gauge(self, name: str) -> float:
        """Read a gauge value."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    # ------------------------------------------------------------------ #
    #  Label
    # ------------------------------------------------------------------ #

    def label(self, name: str, value: str) -> None:
        """Set a string label metric."""
        with self._lock:
            self._labels[name] = value
            self._timestamps[name] = time.monotonic()

    def get_label(self, name: str) -> str:
        """Read a label value."""
        with self._lock:
            return self._labels.get(name, "")

    # ------------------------------------------------------------------ #
    #  Histogram
    # ------------------------------------------------------------------ #

    def histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = HistogramBucket()
            self._timestamps[name] = time.monotonic()
        # Record outside main lock (HistogramBucket has its own lock)
        self._histograms[name].record(value)

    def get_histogram(self, name: str) -> Optional[HistogramBucket]:
        """Get the histogram bucket for a metric."""
        with self._lock:
            return self._histograms.get(name)

    # ------------------------------------------------------------------ #
    #  Timer context manager
    # ------------------------------------------------------------------ #

    class _Timer:
        """Context manager that records elapsed time to a histogram."""

        def __init__(self, collector: MetricsCollector, name: str):
            self._collector = collector
            self._name = name
            self._start = 0.0

        def __enter__(self):
            self._start = time.monotonic()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.monotonic() - self._start) * 1000.0
            self._collector.histogram(self._name, elapsed_ms)

    def timer(self, name: str) -> _Timer:
        """Context manager that records elapsed time (ms) to a histogram.

        Usage::

            with metrics.timer("stt.latency_ms"):
                transcript = engine.transcribe(audio)
        """
        return self._Timer(self, name)

    # ------------------------------------------------------------------ #
    #  Snapshot / Export
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """Return a snapshot of all metrics as a plain dict.

        Useful for dashboard rendering and JSON export.
        """
        with self._lock:
            result = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "labels": dict(self._labels),
                "histograms": {},
            }
            for name, bucket in self._histograms.items():
                result["histograms"][name] = {
                    "mean": bucket.mean,
                    "recent_mean": bucket.recent_mean,
                    "p50": bucket.p50,
                    "p95": bucket.p95,
                    "p99": bucket.p99,
                    "min": bucket.min_val,
                    "max": bucket.max_val,
                    "count": bucket.total_count,
                }
            return result

    def reset(self) -> None:
        """Reset all metrics.  Used in testing."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._labels.clear()
            self._histograms.clear()
            self._timestamps.clear()


# ---------------------------------------------------------------------- #
#  Global singleton
# ---------------------------------------------------------------------- #

_global_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    """Get the global MetricsCollector singleton (lazy-initialised)."""
    global _global_collector
    if _global_collector is None:
        with _collector_lock:
            if _global_collector is None:
                _global_collector = MetricsCollector()
    return _global_collector
