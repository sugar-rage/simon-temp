"""Detection dataclass — universal object detection container.

Replaces the legacy ``logic/detection_schema.py`` with a richer, typed
dataclass that supports spatial reasoning, face identity, OCR text, depth,
and priority metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.models.enums import DetectionSource


@dataclass
class Detection:
    """A single detected object in a frame.

    Fields are progressively enriched as the detection flows through the
    vision pipeline:
      1. YOLO detector fills ``cls``, ``bbox``, ``confidence``, ``source``.
      2. DetectionTracker fills ``track_id``.
      3. FaceRecognizer fills ``face_id`` (for person detections).
      4. SpatialReasoner fills ``position``, ``distance``.
      5. PriorityEngine fills ``priority``.
      6. DepthEstimator fills ``depth_m`` (if enabled).
    """

    # --- Set by detector ---
    cls: str
    """Class name, e.g. ``"car"``, ``"person"``, ``"bicycle"``."""

    bbox: tuple[int, int, int, int]
    """Bounding box ``(x1, y1, x2, y2)`` in pixel coordinates."""

    confidence: float
    """Detection confidence in ``[0.0, 1.0]``."""

    source: str = DetectionSource.YOLO.value
    """Which detector produced this (``"yolo"``, ``"face"``, ``"ocr"``)."""

    # --- Set by tracker ---
    track_id: Optional[int] = None
    """Persistent ID across frames (from ``DetectionTracker``)."""

    # --- Set by face recognizer ---
    face_id: Optional[str] = None
    """Recognized face name, or ``None`` if not a face / unknown."""
    
    face_bbox: Optional[tuple[int, int, int, int]] = None
    """Actual bounding box of the face within the detection."""
    
    face_embedding: Optional[Any] = None
    """Face embedding from InsightFace for registration."""

    # --- Set by OCR ---
    text: Optional[str] = None
    """Recognized text content, if this detection is a text region."""

    # --- Set by spatial reasoner ---
    position: Optional[str] = None
    """Relative position: ``"left"``, ``"center"``, ``"right"``."""

    distance: Optional[str] = None
    """Estimated distance: ``"near"``, ``"medium"``, ``"far"``."""

    # --- Set by priority engine ---
    priority: Optional[int] = None
    """Priority score (lower = more important)."""

    # --- Set by depth estimator ---
    depth_m: Optional[float] = None
    """Estimated depth in meters (from monocular depth estimation)."""

    # --- Convenience properties ---

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Return the area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def iou(self, other: Detection) -> float:
        """Compute Intersection over Union with another detection."""
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        union = self.area + other.area - intersection
        if union == 0:
            return 0.0
        return intersection / union
