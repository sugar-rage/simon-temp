"""Shared fixtures for vision tests."""

from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from core.models.detection import Detection
from core.models.frame import FrameData
from core.events.event_bus import EventBus
from core.capabilities.registry import CapabilityRegistry
from core.metrics.collector import SystemMetricsCollector
from vision.camera.base import BaseCameraSource
from vision.detection.base import BaseDetector
from vision.detection.tracker import DetectionTracker
from vision.ocr.base import BaseOCREngine, OCRResult
from vision.face.base import BaseFaceRecognizer, FaceResult
from vision.scene.scene_analyzer import SceneAnalyzer
from vision.pipeline.perception_fusion import PerceptionFusion


# ── Mock Camera ──────────────────────────────────────────────────────


class MockCamera(BaseCameraSource):
    """Mock camera returning preset frames."""

    def __init__(self, frames: Optional[list] = None) -> None:
        self._frames = frames or []
        self._index = 0
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        return True

    def read(self) -> Optional[FrameData]:
        if not self._opened or not self._frames:
            return None
        if self._index >= len(self._frames):
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened


# ── Mock Detector ────────────────────────────────────────────────────


class MockDetector(BaseDetector):
    """Mock detector returning preset detections."""

    def __init__(self, detections: Optional[list[list[Detection]]] = None) -> None:
        self._detections = detections or []
        self._call_count = 0
        self._ready = True

    def detect(self, frame: Any) -> list[Detection]:
        if not self._detections:
            return []
        result = self._detections[self._call_count % len(self._detections)]
        self._call_count += 1
        return result

    def is_ready(self) -> bool:
        return self._ready


# ── Mock OCR Engine ──────────────────────────────────────────────────


class MockOCREngine(BaseOCREngine):
    """Mock OCR engine returning preset results."""

    def __init__(self, results: Optional[list[OCRResult]] = None) -> None:
        self._results = results or []
        self._ready = True

    def read_text(self, frame: Any) -> list[OCRResult]:
        return list(self._results)

    def is_ready(self) -> bool:
        return self._ready


# ── Mock Face Recognizer ─────────────────────────────────────────────


class MockFaceRecognizer(BaseFaceRecognizer):
    """Mock face recognizer returning preset results."""

    def __init__(self, results: Optional[list[FaceResult]] = None) -> None:
        self._results = results or []
        self._ready = True

    def detect_faces(self, frame: Any) -> list[FaceResult]:
        return list(self._results)

    def is_ready(self) -> bool:
        return self._ready

    def save_face(self, frame: Any, face: FaceResult, name: str) -> bool:
        return True


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def detection_tracker():
    return DetectionTracker(iou_threshold=0.3, max_age_s=2.0)


@pytest.fixture
def scene_analyzer():
    return SceneAnalyzer()


@pytest.fixture
def perception_fusion():
    return PerceptionFusion(
        tracker=DetectionTracker(),
        frame_width=640,
    )


@pytest.fixture
def event_bus():
    bus = EventBus(enable_metrics=False)
    yield bus
    bus.clear()


@pytest.fixture
def capability_registry():
    return CapabilityRegistry()


@pytest.fixture(autouse=True)
def reset_metrics():
    yield
    SystemMetricsCollector.reset_instance()


def make_detection(
    cls: str = "car",
    bbox: tuple[int, int, int, int] = (100, 100, 200, 200),
    confidence: float = 0.9,
    **kwargs,
) -> Detection:
    """Helper to create test detections."""
    return Detection(cls=cls, bbox=bbox, confidence=confidence, **kwargs)


def make_frame(frame_id: int = 1, data: Any = None) -> FrameData:
    """Helper to create test frames."""
    import numpy as np

    if data is None:
        data = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(data=data, frame_id=frame_id)
