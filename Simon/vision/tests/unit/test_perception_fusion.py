"""Unit tests for PerceptionFusion — WorldModel, Entity, events."""

from __future__ import annotations

import pytest

from core.models.detection import Detection
from core.models.enums import Priority, HazardLevel
from core.events import event_types
from vision.ocr.base import OCRResult
from vision.face.base import FaceResult
from vision.scene.scene_analyzer import SceneAnalysis
from vision.pipeline.perception_fusion import (
    PerceptionFusion,
    WorldModel,
    Entity,
)
from vision.detection.tracker import DetectionTracker
from vision.tests.conftest import make_detection


class TestWorldModel:
    def test_empty_world(self):
        world = WorldModel()
        assert world.entity_count == 0
        assert not world.has_hazards

    def test_entities_by_class(self):
        world = WorldModel(entities=[
            Entity(track_id=1, cls="car", bbox=(0, 0, 100, 100), confidence=0.9),
            Entity(track_id=2, cls="person", bbox=(200, 200, 300, 300), confidence=0.8),
            Entity(track_id=3, cls="car", bbox=(400, 400, 500, 500), confidence=0.7),
        ])
        cars = world.get_entities_by_class("car")
        assert len(cars) == 2

    def test_has_hazards(self):
        world = WorldModel(entities=[
            Entity(
                track_id=1, cls="car", bbox=(0, 0, 100, 100),
                confidence=0.9, hazard_level=HazardLevel.CRITICAL,
            ),
        ])
        assert world.has_hazards

    def test_nearest_hazard(self):
        world = WorldModel(entities=[
            Entity(
                track_id=1, cls="car", bbox=(0, 0, 100, 100),
                confidence=0.9, hazard_level=HazardLevel.WARNING,
            ),
            Entity(
                track_id=2, cls="truck", bbox=(200, 200, 300, 300),
                confidence=0.8, hazard_level=HazardLevel.CRITICAL,
            ),
        ])
        nearest = world.get_nearest_hazard()
        assert nearest is not None
        assert nearest.cls == "truck"


class TestPerceptionFusionBasic:
    def test_fuse_empty_detections(self, perception_fusion):
        world = perception_fusion.fuse(detections=[])
        assert world.entity_count == 0

    def test_fuse_with_detections(self, perception_fusion):
        dets = [
            make_detection(cls="car", bbox=(100, 100, 200, 200)),
            make_detection(cls="person", bbox=(300, 300, 400, 400)),
        ]
        world = perception_fusion.fuse(detections=dets)
        assert world.entity_count == 2

    def test_entities_have_track_ids(self, perception_fusion):
        dets = [make_detection(cls="car")]
        world = perception_fusion.fuse(detections=dets)
        assert all(e.track_id > 0 for e in world.entities)


class TestPerceptionFusionSpatial:
    def test_left_position(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(bbox=(10, 100, 50, 200))]  # far left
        world = fusion.fuse(detections=dets)
        assert world.entities[0].position == "left"

    def test_center_position(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(bbox=(250, 100, 350, 200))]  # center
        world = fusion.fuse(detections=dets)
        assert world.entities[0].position == "center"

    def test_right_position(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(bbox=(500, 100, 600, 200))]  # right
        world = fusion.fuse(detections=dets)
        assert world.entities[0].position == "right"


class TestPerceptionFusionDistance:
    def test_near_detection(self):
        fusion = PerceptionFusion(frame_width=640)
        # Large bbox → near
        dets = [make_detection(bbox=(0, 0, 400, 400))]
        world = fusion.fuse(detections=dets)
        assert world.entities[0].distance == "near"

    def test_far_detection(self):
        fusion = PerceptionFusion(frame_width=640)
        # Small bbox → far
        dets = [make_detection(bbox=(300, 300, 320, 320))]
        world = fusion.fuse(detections=dets)
        assert world.entities[0].distance == "far"


class TestPerceptionFusionHazards:
    def test_vehicle_near_is_critical(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(cls="car", bbox=(0, 0, 400, 400))]  # large → near
        world = fusion.fuse(detections=dets)
        assert world.entities[0].hazard_level == HazardLevel.CRITICAL

    def test_vehicle_far_is_caution(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(cls="car", bbox=(300, 300, 310, 310))]  # small → far
        world = fusion.fuse(detections=dets)
        assert world.entities[0].hazard_level == HazardLevel.CAUTION

    def test_non_vehicle_no_hazard(self, perception_fusion):
        dets = [make_detection(cls="person", bbox=(100, 100, 200, 200))]
        world = perception_fusion.fuse(detections=dets)
        assert world.entities[0].hazard_level is None


class TestPerceptionFusionFaces:
    def test_face_correlated_to_person(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(cls="person", bbox=(100, 100, 300, 400))]
        faces = [FaceResult(name="Alice", confidence=0.8, bbox=(120, 110, 280, 250))]
        world = fusion.fuse(detections=dets, faces=faces)
        assert world.entities[0].face_name == "Alice"

    def test_unknown_face_correlated_as_unknown(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(cls="person", bbox=(100, 100, 300, 400))]
        faces = [FaceResult(name="Unknown", confidence=0.1, bbox=(120, 110, 280, 250))]
        world = fusion.fuse(detections=dets, faces=faces)
        assert world.entities[0].face_name == "Unknown"


class TestPerceptionFusionOCR:
    def test_ocr_results_in_world(self, perception_fusion):
        ocr = [OCRResult(text="STOP", confidence=0.9, bbox=(100, 100, 200, 130))]
        world = perception_fusion.fuse(detections=[], ocr_results=ocr)
        assert len(world.ocr_texts) == 1
        assert world.ocr_texts[0].text == "STOP"


class TestPerceptionFusionScene:
    def test_scene_in_world(self, perception_fusion):
        scene = SceneAnalysis(scene_type="street", confidence=0.8)
        world = perception_fusion.fuse(detections=[], scene=scene)
        assert world.scene is not None
        assert world.scene.scene_type == "street"


class TestPerceptionFusionEvents:
    def test_world_update_event_generated(self, perception_fusion):
        dets = [make_detection(cls="person")]
        world = perception_fusion.fuse(detections=dets)
        events = perception_fusion.get_events(world)
        topics = [e.topic for e in events]
        assert event_types.VISION_WORLD_UPDATE in topics

    def test_hazard_event_generated(self):
        fusion = PerceptionFusion(frame_width=640)
        dets = [make_detection(cls="car", bbox=(0, 0, 400, 400))]  # near car
        world = fusion.fuse(detections=dets)
        events = fusion.get_events(world)
        hazard_events = [e for e in events if e.topic == event_types.VISION_HAZARD_DETECTED]
        assert len(hazard_events) == 1
        assert hazard_events[0].is_safety_critical

    def test_no_hazard_event_for_person(self, perception_fusion):
        dets = [make_detection(cls="person")]
        world = perception_fusion.fuse(detections=dets)
        events = perception_fusion.get_events(world)
        hazard_events = [e for e in events if e.topic == event_types.VISION_HAZARD_DETECTED]
        assert len(hazard_events) == 0


class TestPerceptionFusionOCRMerger:
    def test_text_merger_iou(self):
        from vision.ocr.text_merger import TextMerger

        merger = TextMerger(iou_threshold=0.5, cooldown_s=0.0)
        results = [
            OCRResult(text="STOP", confidence=0.9, bbox=(100, 100, 200, 130)),
            OCRResult(text="SIGN", confidence=0.8, bbox=(100, 100, 200, 130)),  # overlapping
        ]
        merged = merger.merge(results)
        assert len(merged) == 1
        assert "STOP" in merged[0].text

    def test_text_merger_cooldown(self):
        from vision.ocr.text_merger import TextMerger

        merger = TextMerger(cooldown_s=10.0)
        results = [OCRResult(text="HELLO", confidence=0.9)]
        first = merger.merge(results)
        second = merger.merge(results)
        assert len(first) == 1
        assert len(second) == 0  # still on cooldown

    def test_text_merger_no_overlap(self):
        from vision.ocr.text_merger import TextMerger

        merger = TextMerger(cooldown_s=0.0)
        results = [
            OCRResult(text="EXIT", confidence=0.9, bbox=(10, 10, 50, 30)),
            OCRResult(text="STOP", confidence=0.8, bbox=(400, 400, 450, 430)),
        ]
        merged = merger.merge(results)
        assert len(merged) == 2


class TestCameraManager:
    def test_mock_camera(self):
        from vision.tests.conftest import MockCamera, make_frame
        camera = MockCamera(frames=[make_frame(1), make_frame(2)])
        assert camera.open()
        f1 = camera.read()
        assert f1 is not None
        assert f1.frame_id == 1
        f2 = camera.read()
        assert f2.frame_id == 2
        camera.release()
        assert not camera.is_opened()
