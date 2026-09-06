"""Perception Fusion — merges all vision results into a unified WorldModel.

This is the single source of truth for "what does SIMON see right now."

See implementation_plan.md Section 4.6 for the full design.
See TDR-005 for the decision rationale.

Responsibilities:
1. Receive per-frame results from YOLO, Face, OCR, Scene.
2. Spatially correlate detections (e.g. face → person bbox).
3. Assign track IDs via DetectionTracker.
4. Merge into a unified WorldModel with Entity objects.
5. Detect enter/exit events for announcement triggers.
6. Emit unified events: vision.world_update, vision.entity_entered,
   vision.entity_exited, vision.hazard_detected.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.models.detection import Detection
from core.models.enums import Priority, HazardLevel
from core.models.events import Event
from core.events import event_types
from vision.ocr.base import OCRResult
from vision.face.base import FaceResult
from vision.scene.scene_analyzer import SceneAnalysis
from vision.detection.tracker import DetectionTracker

logger = logging.getLogger("simon.vision.fusion")


@dataclass
class Entity:
    """A unified entity in the world model.

    Represents a single object/person/text that SIMON is aware of,
    with all enrichment from the vision pipeline.
    """

    track_id: int
    cls: str
    bbox: tuple[int, int, int, int]
    confidence: float
    position: Optional[str] = None       # "left", "center", "right"
    distance: Optional[str] = None       # "near", "medium", "far"
    face_name: Optional[str] = None      # recognized identity
    face_bbox: Optional[tuple[int, int, int, int]] = None
    face_embedding: Optional[Any] = None
    text: Optional[str] = None           # OCR text
    hazard_level: Optional[int] = None   # HazardLevel value
    depth_m: Optional[float] = None
    frames_tracked: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class WorldModel:
    """Unified per-frame world model — output of Perception Fusion.

    This is the single source of truth for what SIMON currently sees.
    """

    entities: list[Entity] = field(default_factory=list)
    scene: Optional[SceneAnalysis] = None
    ocr_texts: list[OCRResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def has_hazards(self) -> bool:
        return any(
            e.hazard_level is not None and e.hazard_level <= HazardLevel.WARNING
            for e in self.entities
        )

    def get_entities_by_class(self, cls: str) -> list[Entity]:
        return [e for e in self.entities if e.cls == cls]

    def get_nearest_hazard(self) -> Optional[Entity]:
        hazards = [
            e for e in self.entities
            if e.hazard_level is not None and e.hazard_level <= HazardLevel.WARNING
        ]
        if not hazards:
            return None
        return min(hazards, key=lambda e: e.hazard_level or 999)


class PerceptionFusion:
    """Merges vision pipeline outputs into a unified WorldModel.

    Parameters
    ----------
    tracker : DetectionTracker
        Cross-frame detection tracker.
    frame_width : int
        Frame width for spatial position calculation.
    vehicle_classes : list[str]
        YOLO class names considered vehicles (for hazard classification).
    """

    def __init__(
        self,
        tracker: Optional[DetectionTracker] = None,
        frame_width: int = 640,
        vehicle_classes: Optional[list[str]] = None,
    ) -> None:
        self._tracker = tracker or DetectionTracker()
        self._frame_width = frame_width
        self._vehicle_classes = set(
            vehicle_classes or ["car", "truck", "bus", "motorcycle", "bicycle"]
        )
        self._previous_track_ids: set[int] = set()
        self._face_cache: dict[int, dict] = {}

    def fuse(
        self,
        detections: list[Detection],
        faces: Optional[list[FaceResult]] = None,
        ocr_results: Optional[list[OCRResult]] = None,
        scene: Optional[SceneAnalysis] = None,
        frame_id: int = 0,
    ) -> WorldModel:
        """Fuse all vision results into a WorldModel.

        Parameters
        ----------
        detections : list[Detection]
            YOLO detection results.
        faces : list[FaceResult], optional
            Face recognition results.
        ocr_results : list[OCRResult], optional
            OCR text results.
        scene : SceneAnalysis, optional
            Scene classification result.
        frame_id : int
            Current frame ID.

        Returns
        -------
        WorldModel
        """
        faces = faces or []
        ocr_results = ocr_results or []

        # Step 1: Track detections across frames
        tracked_detections = self._tracker.update(detections)

        # Step 2: Assign spatial positions
        for det in tracked_detections:
            det.position = self._compute_position(det)
            det.distance = self._estimate_distance(det)

        # Step 3: Correlate faces with person detections
        self._correlate_faces(tracked_detections, faces)

        # Step 4: Classify hazards
        for det in tracked_detections:
            if det.cls in self._vehicle_classes:
                det.priority = Priority.OBSTACLE

        # Step 5: Build entities
        entities = self._build_entities(tracked_detections)

        # Step 6: Detect enter/exit events
        current_ids = {e.track_id for e in entities}
        entered = current_ids - self._previous_track_ids
        exited = self._previous_track_ids - current_ids
        self._previous_track_ids = current_ids

        world = WorldModel(
            entities=entities,
            scene=scene,
            ocr_texts=ocr_results,
            timestamp=time.time(),
            frame_id=frame_id,
        )

        # Step 8: Clean up stale tracks
        # self._tracker.cleanup()

        return world

    def get_events(self, world: WorldModel) -> list[Event]:
        """Generate events from a WorldModel for the Event Bus.

        Returns a list of events to publish.
        """
        events: list[Event] = []

        # World update event
        events.append(Event(
            topic=event_types.VISION_WORLD_UPDATE,
            data={
                "entity_count": world.entity_count,
                "scene_type": world.scene.scene_type if world.scene else "unknown",
                "frame_id": world.frame_id,
                "world": world,
            },
            priority=Priority.INFORMATIONAL,
            source="vision.fusion",
        ))

        # Hazard events (safety-critical)
        for entity in world.entities:
            if (
                entity.hazard_level is not None
                and entity.hazard_level <= HazardLevel.WARNING
            ):
                events.append(Event(
                    topic=event_types.VISION_HAZARD_DETECTED,
                    data={
                        "cls": entity.cls,
                        "track_id": entity.track_id,
                        "position": entity.position,
                        "distance": entity.distance,
                        "hazard_level": entity.hazard_level,
                    },
                    priority=Priority.SAFETY_CRITICAL,
                    source="vision.fusion",
                ))

        # OCR Warnings
        warning_keywords = {"warning", "danger", "caution", "stop", "hazard", "fire", "no entry", "restricted", "emergency"}
        for ocr in world.ocr_texts:
            text_lower = ocr.text.lower()
            if any(kw in text_lower for kw in warning_keywords):
                events.append(Event(
                    topic=event_types.VISION_TEXT_WARNING,
                    data={
                        "text": ocr.text,
                        "priority": Priority.SAFETY_CRITICAL,
                    },
                    priority=Priority.SAFETY_CRITICAL,
                    source="vision.fusion",
                ))

        return events

    def _compute_position(self, det: Detection) -> str:
        """Determine relative horizontal position (left / slightly_left / center / slightly_right / right)."""
        cx = det.center[0]
        w = float(self._frame_width)
        if cx < 0.2 * w:
            return "left"
        elif cx < 0.4 * w:
            return "slightly_left"
        elif cx < 0.6 * w:
            return "center"
        elif cx < 0.8 * w:
            return "slightly_right"
        else:
            return "right"

    def _estimate_distance(self, det: Detection) -> str:
        """Estimate distance based on bounding box area heuristic."""
        area_ratio = det.area / (self._frame_width * self._frame_width)
        if area_ratio > 0.15:
            return "near"
        elif area_ratio > 0.03:
            return "medium"
        else:
            return "far"

    def _correlate_faces(
        self,
        detections: list[Detection],
        faces: list[FaceResult],
    ) -> None:
        """Match face results to person detections by bbox overlap, with tolerance."""
        person_dets = [d for d in detections if d.cls == "person"]
        now = time.time()
        
        # 1. Match current faces to person detections
        matched_tracks = set()
        for face in faces:
            best_det = None
            best_iou = 0.0

            for det in person_dets:
                face_det = Detection(
                    cls="face", bbox=face.bbox, confidence=face.confidence
                )
                iou = det.iou(face_det)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou > 0.1:
                best_det.face_id = face.name
                best_det.face_bbox = face.bbox
                best_det.face_embedding = face.embedding
                
                if best_det.track_id is not None:
                    if best_det.track_id in self._face_cache:
                        logger.debug(f"[FACE_TRACK] matched existing track_id={best_det.track_id}")
                    else:
                        logger.info(f"[FACE_TRACK] new track_id={best_det.track_id}")
                        
                    self._face_cache[best_det.track_id] = {
                        "face_id": face.name,
                        "face_bbox": face.bbox,
                        "face_embedding": face.embedding,
                        "timestamp": now
                    }
                    matched_tracks.add(best_det.track_id)

        # 2. Apply cached faces for unmatched persons within tolerance (0.5s)
        for det in person_dets:
            if det.track_id is not None and det.track_id not in matched_tracks:
                if det.track_id in self._face_cache:
                    cached = self._face_cache[det.track_id]
                    if now - cached["timestamp"] < 0.5:
                        det.face_id = cached["face_id"]
                        det.face_bbox = cached["face_bbox"]
                        det.face_embedding = cached["face_embedding"]
                        
        # 3. Clean up old cache entries
        stale_keys = [tid for tid, data in self._face_cache.items() if now - data["timestamp"] >= 0.5]
        for tid in stale_keys:
            del self._face_cache[tid]

    def _build_entities(
        self, detections: list[Detection]
    ) -> list[Entity]:
        """Convert tracked detections to Entity objects."""
        entities: list[Entity] = []
        for det in detections:
            hazard = None
            if det.cls in self._vehicle_classes:
                if det.distance == "near":
                    hazard = HazardLevel.CRITICAL
                elif det.distance == "medium":
                    hazard = HazardLevel.WARNING
                else:
                    hazard = HazardLevel.CAUTION

            entities.append(Entity(
                track_id=det.track_id or 0,
                cls=det.cls,
                bbox=det.bbox,
                confidence=det.confidence,
                position=det.position,
                distance=det.distance,
                face_name=det.face_id,
                face_bbox=det.face_bbox,
                face_embedding=det.face_embedding,
                text=det.text,
                hazard_level=hazard,
                depth_m=det.depth_m,
            ))

        return entities
