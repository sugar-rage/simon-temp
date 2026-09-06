"""YOLOv8 object detector — wraps Ultralytics YOLO.

Refactored from ``perception/yolo_detector.py`` with config-driven parameters,
proper error handling, and Detection dataclass output.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.models.detection import Detection
from core.config.system_config import DetectionConfig
from core.errors.exceptions import ModelLoadError, InferenceError
from vision.detection.base import BaseDetector

logger = logging.getLogger("simon.vision.detection")


class YOLODetector(BaseDetector):
    """YOLOv8 detector using Ultralytics.

    Parameters
    ----------
    config : DetectionConfig
        Detection configuration (model path, thresholds, device).
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self._config = config or DetectionConfig()
        self._model: Any = None
        self._class_names: dict[int, str] = {}
        self._ready = False

    def load(self) -> None:
        """Load the YOLO model. Raises ModelLoadError on failure."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ModelLoadError(
                self._config.model,
                reason="ultralytics package not installed",
            )

        try:
            self._model = YOLO(self._config.model)
            self._class_names = self._model.names or {}
            self._ready = True
            logger.info(
                "YOLO model loaded: %s (%d classes)",
                self._config.model,
                len(self._class_names),
            )
        except Exception as e:
            self._ready = False
            raise ModelLoadError(self._config.model, reason=str(e))

    def detect(self, frame: Any) -> list[Detection]:
        """Run YOLO detection on a frame.

        Returns a list of Detection dataclasses.
        """
        if not self._ready or self._model is None:
            return []

        try:
            results = self._model(
                frame,
                conf=self._config.confidence_threshold,
                iou=self._config.nms_threshold,
                verbose=False,
                device=self._config.device if self._config.device != "auto" else None,
            )[0]
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            raise InferenceError(self._config.model, reason=str(e))

        detections: list[Detection] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self._class_names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                Detection(
                    cls=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    source="yolo",
                )
            )

        # Limit max detections
        if len(detections) > self._config.max_detections:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[: self._config.max_detections]

        return detections

    def is_ready(self) -> bool:
        return self._ready

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)
