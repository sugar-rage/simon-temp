"""Unit tests for core data models — Detection, FrameData, GeoLocation, Event, Action."""

from __future__ import annotations

import time

import pytest

from core.models.detection import Detection
from core.models.frame import FrameData
from core.models.location import GeoLocation, RouteStep, Route, Waypoint
from core.models.events import Event
from core.models.actions import Action, ActionResult
from core.models.enums import (
    Priority,
    SubsystemStatus,
    HazardLevel,
    NavigationManeuver,
    TaskStatus,
)


# ── Detection Tests ──────────────────────────────────────────────────


class TestDetection:
    def test_basic_creation(self):
        d = Detection(cls="car", bbox=(10, 20, 100, 200), confidence=0.85)
        assert d.cls == "car"
        assert d.confidence == 0.85
        assert d.source == "yolo"

    def test_center_calculation(self):
        d = Detection(cls="person", bbox=(0, 0, 100, 200), confidence=0.9)
        assert d.center == (50, 100)

    def test_area_calculation(self):
        d = Detection(cls="car", bbox=(10, 20, 110, 220), confidence=0.8)
        assert d.area == 100 * 200

    def test_width_height(self):
        d = Detection(cls="car", bbox=(10, 20, 60, 120), confidence=0.7)
        assert d.width == 50
        assert d.height == 100

    def test_iou_identical(self):
        d1 = Detection(cls="car", bbox=(0, 0, 100, 100), confidence=0.9)
        d2 = Detection(cls="car", bbox=(0, 0, 100, 100), confidence=0.8)
        assert d1.iou(d2) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        d1 = Detection(cls="car", bbox=(0, 0, 50, 50), confidence=0.9)
        d2 = Detection(cls="car", bbox=(100, 100, 150, 150), confidence=0.8)
        assert d1.iou(d2) == 0.0

    def test_iou_partial_overlap(self):
        d1 = Detection(cls="car", bbox=(0, 0, 100, 100), confidence=0.9)
        d2 = Detection(cls="car", bbox=(50, 50, 150, 150), confidence=0.8)
        intersection = 50 * 50  # 2500
        union = 10000 + 10000 - 2500  # 17500
        assert d1.iou(d2) == pytest.approx(intersection / union)

    def test_optional_fields_default_none(self):
        d = Detection(cls="person", bbox=(0, 0, 50, 50), confidence=0.5)
        assert d.track_id is None
        assert d.face_id is None
        assert d.text is None
        assert d.position is None
        assert d.distance is None
        assert d.priority is None
        assert d.depth_m is None


# ── FrameData Tests ──────────────────────────────────────────────────


class TestFrameData:
    def test_basic_creation(self):
        frame = FrameData(data=None, frame_id=1)
        assert frame.frame_id == 1
        assert frame.resolution == (640, 480)

    def test_width_height_properties(self):
        frame = FrameData(data=None, resolution=(1920, 1080))
        assert frame.width == 1920
        assert frame.height == 1080

    def test_is_valid_with_none(self):
        frame = FrameData(data=None)
        assert not frame.is_valid

    def test_timestamp_auto_set(self):
        before = time.time()
        frame = FrameData(data=None)
        after = time.time()
        assert before <= frame.timestamp <= after


# ── GeoLocation Tests ────────────────────────────────────────────────


class TestGeoLocation:
    def test_basic_creation(self):
        loc = GeoLocation(latitude=40.7128, longitude=-74.0060)
        assert loc.latitude == pytest.approx(40.7128)

    def test_distance_to_same_point(self):
        loc = GeoLocation(latitude=40.0, longitude=-74.0)
        assert loc.distance_to(loc) == pytest.approx(0.0, abs=0.01)

    def test_distance_to_known(self):
        # NYC to LA: ~3944 km
        nyc = GeoLocation(latitude=40.7128, longitude=-74.0060)
        la = GeoLocation(latitude=34.0522, longitude=-118.2437)
        dist = nyc.distance_to(la)
        assert 3_900_000 < dist < 4_000_000  # meters


# ── Event Tests ──────────────────────────────────────────────────────


class TestEvent:
    def test_basic_creation(self):
        e = Event(topic="test.event")
        assert e.topic == "test.event"
        assert e.source == "system"
        assert len(e.correlation_id) == 12

    def test_safety_critical_auto_flag(self):
        e = Event(topic="safety.hazard", priority=Priority.EMERGENCY)
        assert e.is_safety_critical

    def test_non_safety_not_flagged(self):
        e = Event(topic="vision.detection", priority=Priority.INFORMATIONAL)
        assert not e.is_safety_critical

    def test_ordering_by_priority(self):
        e1 = Event(topic="a", priority=Priority.EMERGENCY)
        e2 = Event(topic="b", priority=Priority.LOW)
        assert e1 < e2

    def test_derive_preserves_correlation_id(self):
        original = Event(topic="vision.detection", source="vision")
        derived = original.derive("safety.hazard", source="safety")
        assert derived.correlation_id == original.correlation_id
        assert derived.topic == "safety.hazard"
        assert derived.source == "safety"


# ── RouteStep Tests ──────────────────────────────────────────────────


class TestRouteStep:
    def test_maneuver_enum(self):
        step = RouteStep(
            instruction="Turn left",
            distance_m=50.0,
            maneuver="turn-left",
        )
        assert step.maneuver_enum == NavigationManeuver.TURN_LEFT

    def test_invalid_maneuver_defaults(self):
        step = RouteStep(
            instruction="Do something",
            distance_m=10.0,
            maneuver="unknown",
        )
        assert step.maneuver_enum == NavigationManeuver.STRAIGHT


# ── Enum Tests ───────────────────────────────────────────────────────


class TestEnums:
    def test_priority_safety_critical(self):
        assert Priority.EMERGENCY.is_safety_critical
        assert Priority.SAFETY_CRITICAL.is_safety_critical
        assert not Priority.OBSTACLE.is_safety_critical

    def test_priority_should_interrupt(self):
        assert Priority.EMERGENCY.should_interrupt
        assert Priority.OBSTACLE.should_interrupt
        assert not Priority.NAVIGATION.should_interrupt

    def test_hazard_level_urgent(self):
        assert HazardLevel.CRITICAL.is_urgent
        assert HazardLevel.WARNING.is_urgent
        assert not HazardLevel.CAUTION.is_urgent

    def test_subsystem_status_values(self):
        assert SubsystemStatus.RUNNING.name == "RUNNING"
        assert SubsystemStatus.ERROR.name == "ERROR"

    def test_task_status_values(self):
        assert TaskStatus.PENDING.name == "PENDING"
        assert TaskStatus.COMPLETED.name == "COMPLETED"


# ── Action Tests ─────────────────────────────────────────────────────


class TestAction:
    def test_basic_creation(self):
        a = Action(action_type="speak", args={"text": "hello"})
        assert a.action_type == "speak"
        assert a.args["text"] == "hello"

    def test_action_result(self):
        a = Action(action_type="navigate")
        r = ActionResult(action=a, success=True, message="Route computed")
        assert r.success
        assert r.message == "Route computed"
