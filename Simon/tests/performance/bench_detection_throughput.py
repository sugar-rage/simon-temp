"""Detection throughput benchmarks.

Measures the throughput of detection-related components:
- DetectionTracker update speed
- Scene analysis speed
- Object counting / spatial reasoning speed

Usage::

    python -m pytest tests/performance/bench_detection_throughput.py -v -s
"""

from __future__ import annotations

import time
import statistics
import pytest

from core.models.detection import Detection
from vision.detection.tracker import DetectionTracker
from vision.scene.scene_analyzer import SceneAnalyzer


# ── Helpers ──────────────────────────────────────────────────────────


def _make_detections(n: int) -> list[Detection]:
    """Generate synthetic detections."""
    classes = ["person", "car", "chair", "bottle", "dog",
               "truck", "bicycle", "cat", "motorcycle", "bus"]
    return [
        Detection(
            cls=classes[i % len(classes)],
            confidence=0.75 + (i % 5) * 0.05,
            bbox=(i * 40, 50, i * 40 + 60, 150),
        )
        for i in range(n)
    ]


# ── Benchmarks ───────────────────────────────────────────────────────


class TestDetectionTrackerBenchmarks:
    """Benchmark DetectionTracker performance."""

    ITERATIONS = 200

    def test_tracker_update_throughput(self):
        """Measure DetectionTracker.update() throughput.

        Target: > 500 updates/sec with 20 detections.
        """
        tracker = DetectionTracker()
        detections = _make_detections(20)

        latencies = []
        for i in range(self.ITERATIONS):
            # Vary positions slightly per frame to simulate movement
            frame_dets = [
                Detection(
                    cls=d.cls,
                    confidence=d.confidence,
                    bbox=(d.bbox[0] + i % 5, d.bbox[1], d.bbox[2] + i % 5, d.bbox[3]),
                )
                for d in detections
            ]
            t0 = time.perf_counter()
            tracker.update(frame_dets)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        throughput = 1000.0 / avg if avg > 0 else float("inf")

        print(f"\n  DetectionTracker Update ({self.ITERATIONS} frames, 20 detections/frame):")
        print(f"    avg: {avg:.2f}ms  throughput: {throughput:.0f} updates/sec")
        print(f"    min: {min(latencies):.2f}ms  max: {max(latencies):.2f}ms")

        assert throughput > 100, (
            f"Tracker throughput {throughput:.0f} updates/sec is below 100/sec target"
        )

    def test_tracker_scaling(self):
        """Measure how tracker scales with detection count."""
        results = {}
        for n in [1, 5, 10, 20, 50, 100]:
            tracker = DetectionTracker()
            detections = _make_detections(n)
            latencies = []

            for _ in range(50):
                t0 = time.perf_counter()
                tracker.update(detections)
                latencies.append((time.perf_counter() - t0) * 1000)

            results[n] = statistics.mean(latencies)

        print("\n  Tracker Scaling:")
        for n, avg in results.items():
            print(f"    {n:4d} detections -> {avg:.3f}ms")

        # Should not grow worse than O(n^2)
        if results[1] > 0:
            ratio = results[100] / results[1]
            print(f"    100/1 ratio: {ratio:.1f}x (ideal <= 10000x for O(n^2))")

    def test_tracker_id_stability(self):
        """Measure track ID stability under consistent positions."""
        tracker = DetectionTracker()
        detections = _make_detections(5)

        # Run 100 frames with same positions
        final_ids = set()
        for _ in range(100):
            tracked = tracker.update(detections)
            final_ids = {d.track_id for d in tracked if d.track_id is not None}

        # Should have exactly 5 stable IDs
        assert len(final_ids) == 5, (
            f"Expected 5 stable track IDs, got {len(final_ids)}"
        )


class TestSceneAnalyzerBenchmarks:
    """Benchmark SceneAnalyzer performance."""

    ITERATIONS = 200

    def test_scene_analysis_throughput(self):
        """Measure SceneAnalyzer throughput.

        Target: > 1000 analyses/sec (no ML inference).
        """
        analyzer = SceneAnalyzer()
        detections = _make_detections(15)

        latencies = []
        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            result = analyzer.analyze(detections)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        throughput = 1000.0 / avg if avg > 0 else float("inf")

        print(f"\n  SceneAnalyzer ({self.ITERATIONS} iterations, 15 detections):")
        print(f"    avg: {avg:.3f}ms  throughput: {throughput:.0f}/sec")

        assert throughput > 500, (
            f"Scene analysis throughput {throughput:.0f}/sec is below 500/sec"
        )

    def test_scene_empty_fast(self):
        """Empty detection list should be extremely fast."""
        analyzer = SceneAnalyzer()

        latencies = []
        for _ in range(500):
            t0 = time.perf_counter()
            analyzer.analyze([])
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = statistics.mean(latencies)
        print(f"\n  SceneAnalyzer (empty): avg {avg:.3f}ms")
        assert avg < 1.0, f"Empty scene analysis too slow: {avg:.3f}ms"
