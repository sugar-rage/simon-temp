"""Frame processing benchmarks.

Measures latency and throughput for the vision pipeline stages:
- Perception fusion (detections → WorldModel)
- Event dispatch from vision events
- WorldModel creation throughput

Target: Frame → YOLO → Fusion → Event → Safety → TTS < 500ms

Usage::

    python -m pytest tests/performance/bench_frame_processing.py -v -s
"""

from __future__ import annotations

import time
import statistics
import pytest

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.models.detection import Detection
from vision.pipeline.perception_fusion import PerceptionFusion, WorldModel


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    bus.start()
    yield bus
    bus.stop()
    bus.clear()


@pytest.fixture
def perception_fusion():
    return PerceptionFusion()


def _make_detections(n: int) -> list[Detection]:
    """Generate synthetic detections for benchmarking."""
    detections = []
    classes = ["person", "car", "chair", "bottle", "dog"]
    for i in range(n):
        detections.append(Detection(
            cls=classes[i % len(classes)],
            confidence=0.8,
            bbox=(i * 50, 100, i * 50 + 80, 200),
        ))
    return detections


# ── Benchmarks ───────────────────────────────────────────────────────


class TestFrameProcessingBenchmarks:
    """Benchmark the perception fusion pipeline."""

    ITERATIONS = 100
    MAX_LATENCY_MS = 10.0  # Fusion only (excludes YOLO inference)

    def test_perception_fusion_latency(self, perception_fusion):
        """Measure perception fusion latency (detections → WorldModel).

        Target: < 10ms per frame (fusion only, not YOLO inference).
        """
        detections = _make_detections(10)
        latencies = []

        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            world = perception_fusion.fuse(
                detections=detections,
                faces=[],
                ocr_results=[],
                frame_id=0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\n  Perception Fusion ({self.ITERATIONS} iterations, 10 detections):")
        print(f"    avg: {avg:.2f}ms  p95: {p95:.2f}ms  p99: {p99:.2f}ms")
        print(f"    min: {min(latencies):.2f}ms  max: {max(latencies):.2f}ms")

        assert avg < self.MAX_LATENCY_MS, (
            f"Average fusion latency {avg:.2f}ms exceeds target {self.MAX_LATENCY_MS}ms"
        )

    def test_perception_fusion_scaling(self, perception_fusion):
        """Measure how fusion scales with detection count."""
        results = {}
        for n_detections in [1, 5, 10, 25, 50]:
            detections = _make_detections(n_detections)
            latencies = []

            for _ in range(50):
                t0 = time.perf_counter()
                perception_fusion.fuse(
                    detections=detections, faces=[], ocr_results=[], frame_id=0,
                )
                latencies.append((time.perf_counter() - t0) * 1000)

            results[n_detections] = statistics.mean(latencies)

        print("\n  Fusion Scaling (detections -> avg latency):")
        for n, avg in results.items():
            print(f"    {n:3d} detections -> {avg:.2f}ms")

        # Should be sub-quadratic: 50x detections should take < 2500x longer
        assert results[50] < results[1] * 2500, "Fusion scaling is worse than O(n^2)"

    def test_event_dispatch_latency(self, event_bus):
        """Measure event bus dispatch latency for vision events.

        Target: < 50ms per event dispatch (async).
        """
        received = []
        event_bus.subscribe(
            event_types.VISION_WORLD_UPDATE,
            lambda e: received.append(time.perf_counter()),
            source="bench",
        )

        latencies = []
        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            event_bus.publish(Event(
                topic=event_types.VISION_WORLD_UPDATE,
                priority=Priority.INFORMATIONAL,
                source="bench",
                data={"frame_id": 0},
            ))
            time.sleep(0.005)
            if received:
                latencies.append((received[-1] - t0) * 1000)

        if latencies:
            avg = statistics.mean(latencies)
            print(f"\n  Event Dispatch Latency ({len(latencies)} samples):")
            print(f"    avg: {avg:.2f}ms")
            assert avg < 50.0, f"Event dispatch too slow: {avg:.2f}ms"

    def test_world_model_creation_throughput(self, perception_fusion):
        """Measure WorldModel creation throughput (frames/sec)."""
        detections = _make_detections(10)

        t0 = time.perf_counter()
        frames = 0
        while (time.perf_counter() - t0) < 1.0:
            perception_fusion.fuse(
                detections=detections, faces=[], ocr_results=[], frame_id=frames,
            )
            frames += 1

        elapsed = time.perf_counter() - t0
        fps = frames / elapsed

        print(f"\n  WorldModel Throughput: {fps:.0f} fusions/sec ({frames} frames in {elapsed:.2f}s)")
        assert fps > 30, f"Fusion throughput {fps:.0f} fps is below 30 fps target"
