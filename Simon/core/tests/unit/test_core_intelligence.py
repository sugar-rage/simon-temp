"""Unit tests for Phase 3 — Core Intelligence.

Tests: PriorityEngine, SpatialReasoner, LanguageGenerator, LogicController,
SafetyEngine, SystemContext, AnnouncementTracker, SessionMemory.
"""

from __future__ import annotations

import time

import pytest

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority, HazardLevel
from core.logic.priority_engine import PriorityEngine
from core.logic.spatial_reasoner import SpatialReasoner
from core.logic.language_generator import LanguageGenerator
from core.logic.logic_controller import LogicController
from core.safety.safety_engine import (
    SafetyEngine, HazardClassifier, SafetyRules, EmergencyHandler, HazardAlert,
)
from core.context.system_context import SystemContext, SpatialContext, NavigationContext
from core.memory.announcement_tracker import AnnouncementTracker
from core.memory.memory_store import SessionMemory, PersistentMemory, MemoryEntry
from core.state.state_machine import SystemState
from core.metrics.collector import SystemMetricsCollector
from vision.pipeline.perception_fusion import Entity, WorldModel
from vision.scene.scene_analyzer import SceneAnalysis
from vision.ocr.base import OCRResult


# ── Helpers ──────────────────────────────────────────────────────────


def make_entity(
    cls: str = "car",
    position: str = "center",
    distance: str = "medium",
    hazard_level: int | None = None,
    face_name: str | None = None,
    track_id: int = 1,
) -> Entity:
    return Entity(
        track_id=track_id,
        cls=cls,
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        position=position,
        distance=distance,
        hazard_level=hazard_level,
        face_name=face_name,
    )


def make_world(entities: list[Entity] | None = None, scene_type: str = "unknown") -> WorldModel:
    return WorldModel(
        entities=entities or [],
        scene=SceneAnalysis(scene_type=scene_type) if scene_type != "none" else None,
    )


@pytest.fixture(autouse=True)
def reset_metrics():
    yield
    SystemMetricsCollector.reset_instance()


@pytest.fixture
def event_bus():
    bus = EventBus(enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


# ═════════════════════════════════════════════════════════════════════
# PriorityEngine Tests
# ═════════════════════════════════════════════════════════════════════


class TestPriorityEngine:
    def test_critical_hazard_gets_emergency(self):
        engine = PriorityEngine()
        entity = make_entity(hazard_level=HazardLevel.CRITICAL)
        assert engine.score(entity) == Priority.EMERGENCY

    def test_warning_hazard_gets_safety_critical(self):
        engine = PriorityEngine()
        entity = make_entity(hazard_level=HazardLevel.WARNING)
        assert engine.score(entity) == Priority.SAFETY_CRITICAL

    def test_known_face_gets_face_priority(self):
        engine = PriorityEngine()
        entity = make_entity(face_name="Alice")
        assert engine.score(entity) == Priority.FACE

    def test_unknown_face_not_boosted(self):
        engine = PriorityEngine()
        entity = make_entity(face_name="Unknown")
        assert engine.score(entity) != Priority.FACE

    def test_near_object_gets_obstacle(self):
        engine = PriorityEngine()
        entity = make_entity(distance="near")
        score = engine.score(entity)
        assert score <= Priority.OBSTACLE

    def test_high_priority_class_boosted(self):
        engine = PriorityEngine()
        car = make_entity(cls="car")
        bench = make_entity(cls="bench")
        assert engine.score(car) <= engine.score(bench)

    def test_rank_entities_sorted(self):
        engine = PriorityEngine()
        world = make_world([
            make_entity(cls="car", hazard_level=HazardLevel.CRITICAL, track_id=1),
            make_entity(cls="person", track_id=2),
            make_entity(cls="car", face_name="Bob", track_id=3),
        ])
        ranked = engine.rank_entities(world)
        # Critical hazard should be first
        assert ranked[0].hazard_level == HazardLevel.CRITICAL

    def test_empty_world_returns_empty(self):
        engine = PriorityEngine()
        assert engine.rank_entities(make_world([])) == []


# ═════════════════════════════════════════════════════════════════════
# SpatialReasoner Tests
# ═════════════════════════════════════════════════════════════════════


class TestSpatialReasoner:
    def test_entity_left_near(self):
        sr = SpatialReasoner()
        desc = sr.describe_entity(make_entity(position="left", distance="near"))
        assert "left" in desc
        assert "nearby" in desc

    def test_entity_center(self):
        sr = SpatialReasoner()
        desc = sr.describe_entity(make_entity(position="center"))
        assert "ahead" in desc

    def test_entity_right(self):
        sr = SpatialReasoner()
        desc = sr.describe_entity(make_entity(position="right"))
        assert "right" in desc

    def test_entity_far(self):
        sr = SpatialReasoner()
        desc = sr.describe_entity(make_entity(distance="far"))
        assert "distance" in desc

    def test_known_face_uses_name(self):
        sr = SpatialReasoner()
        desc = sr.describe_entity(make_entity(face_name="Alice"))
        assert "Alice" in desc

    def test_scene_description(self):
        sr = SpatialReasoner()
        world = make_world([
            make_entity(cls="car", track_id=1),
            make_entity(cls="car", track_id=2),
            make_entity(cls="person", track_id=3),
        ], scene_type="street")
        desc = sr.describe_scene(world)
        assert "street" in desc.lower()
        assert "car" in desc.lower()

    def test_empty_scene(self):
        sr = SpatialReasoner()
        desc = sr.describe_scene(make_world([]))
        assert "nothing" in desc.lower()

    def test_hazard_description_critical(self):
        sr = SpatialReasoner()
        desc = sr.get_hazard_description(
            make_entity(hazard_level=HazardLevel.CRITICAL)
        )
        assert "Warning" in desc

    def test_hazard_description_warning(self):
        sr = SpatialReasoner()
        desc = sr.get_hazard_description(
            make_entity(hazard_level=HazardLevel.WARNING)
        )
        assert "Caution" in desc


# ═════════════════════════════════════════════════════════════════════
# LanguageGenerator Tests
# ═════════════════════════════════════════════════════════════════════


class TestLanguageGenerator:
    def test_announce_entity(self):
        gen = LanguageGenerator()
        text = gen.announce_entity(make_entity(cls="car", position="left"))
        assert text is not None
        assert "car" in text.lower() or "left" in text.lower()

    def test_announce_hazard(self):
        gen = LanguageGenerator()
        text = gen.announce_entity(
            make_entity(cls="truck", hazard_level=HazardLevel.CRITICAL)
        )
        assert "truck" in text.lower()

    def test_announce_face(self):
        gen = LanguageGenerator()
        text = gen.announce_entity(make_entity(face_name="Alice"))
        assert "Alice" in text

    def test_navigation_instruction(self):
        gen = LanguageGenerator()
        text = gen.navigation_instruction("Turn left", 50.0)
        assert "50" in text
        assert "Turn left" in text

    def test_arrival_announcement(self):
        gen = LanguageGenerator()
        text = gen.arrival_announcement("Central Park")
        assert "Central Park" in text

    def test_off_route(self):
        gen = LanguageGenerator()
        text = gen.off_route_announcement(25.0)
        assert "25" in text

    def test_scene_summary(self):
        gen = LanguageGenerator()
        world = make_world([
            make_entity(cls="car", track_id=1),
            make_entity(cls="person", track_id=2),
        ], scene_type="street")
        text = gen.scene_summary(world)
        assert "car" in text.lower()

    def test_scene_summary_empty(self):
        gen = LanguageGenerator()
        text = gen.scene_summary(make_world([]))
        assert "don't" in text.lower() or "nothing" in text.lower()

    def test_capability_announcement(self):
        gen = LanguageGenerator()
        text = gen.capability_announcement(["camera", "detection"], ["ocr"])
        assert "2 features" in text
        assert "ocr" in text.lower()


# ═════════════════════════════════════════════════════════════════════
# LogicController Tests
# ═════════════════════════════════════════════════════════════════════


class TestLogicController:
    def test_process_world_generates_announcements(self, event_bus):
        lc = LogicController(event_bus)
        world = make_world([
            make_entity(cls="car", position="left", track_id=1),
            make_entity(cls="person", position="right", track_id=2),
        ])
        announcements = lc.process_world(world)
        assert len(announcements) > 0

    def test_deduplication(self, event_bus):
        lc = LogicController(event_bus)
        world = make_world([make_entity(cls="car", position="left")])
        first = lc.process_world(world)
        second = lc.process_world(world)
        assert len(first) > 0
        assert len(second) == 0  # deduplicated

    def test_max_announcements_per_frame(self, event_bus):
        lc = LogicController(event_bus, max_announcements_per_frame=2)
        entities = [
            make_entity(cls=f"obj_{i}", position="center", track_id=i)
            for i in range(10)
        ]
        world = make_world(entities)
        announcements = lc.process_world(world)
        assert len(announcements) <= 2

    def test_get_scene_description(self, event_bus):
        lc = LogicController(event_bus)
        lc.process_world(make_world([make_entity(cls="car")], scene_type="street"))
        desc = lc.get_scene_description()
        assert "car" in desc.lower()

    def test_hazard_event_handler(self, event_bus):
        received = []
        event_bus.subscribe(event_types.SAFETY_EMERGENCY, lambda e: received.append(e))
        lc = LogicController(event_bus)
        lc.start()
        event_bus.publish_sync(Event(
            topic=event_types.VISION_HAZARD_DETECTED,
            data={"cls": "truck", "position": "left", "distance": "near", "hazard_level": 1},
            priority=Priority.SAFETY_CRITICAL,
        ))
        assert len(received) == 1


# ═════════════════════════════════════════════════════════════════════
# SafetyEngine Tests
# ═════════════════════════════════════════════════════════════════════


class TestHazardClassifier:
    def test_vehicle_near_is_critical(self):
        hc = HazardClassifier()
        entity = make_entity(cls="car", distance="near")
        assert hc.classify(entity) == HazardLevel.CRITICAL

    def test_vehicle_medium_is_warning(self):
        hc = HazardClassifier()
        entity = make_entity(cls="truck", distance="medium")
        assert hc.classify(entity) == HazardLevel.WARNING

    def test_vehicle_far_is_caution(self):
        hc = HazardClassifier()
        entity = make_entity(cls="bus", distance="far")
        assert hc.classify(entity) == HazardLevel.CAUTION

    def test_non_vehicle_returns_none(self):
        hc = HazardClassifier()
        entity = make_entity(cls="chair", distance="near")
        assert hc.classify(entity) is None


class TestSafetyRules:
    def test_critical_always_alerts(self):
        rules = SafetyRules()
        entity = make_entity()
        assert rules.should_alert(entity, HazardLevel.CRITICAL)
        assert rules.should_alert(entity, HazardLevel.CRITICAL)  # no cooldown

    def test_warning_has_cooldown(self):
        rules = SafetyRules(alert_cooldown_s=10.0)
        entity = make_entity(cls="car", position="left")
        assert rules.should_alert(entity, HazardLevel.WARNING)
        assert not rules.should_alert(entity, HazardLevel.WARNING)  # on cooldown


class TestSafetyEngine:
    def test_evaluate_with_hazards(self):
        engine = SafetyEngine()
        world = make_world([
            make_entity(cls="car", distance="near"),
        ])
        alerts = engine.evaluate(world)
        assert len(alerts) > 0
        assert alerts[0].level == HazardLevel.CRITICAL

    def test_evaluate_no_hazards(self):
        engine = SafetyEngine()
        world = make_world([make_entity(cls="chair")])
        alerts = engine.evaluate(world)
        assert len(alerts) == 0

    def test_disabled_engine(self):
        from core.config.system_config import SafetyConfig
        config = SafetyConfig(enabled=False)
        engine = SafetyEngine(config=config)
        world = make_world([make_entity(cls="car", distance="near")])
        assert engine.evaluate(world) == []

    def test_emergency_handler_called(self, event_bus):
        received = []
        event_bus.subscribe("safety.*", lambda e: received.append(e))
        engine = SafetyEngine(event_bus=event_bus)
        world = make_world([make_entity(cls="car", distance="near")])
        engine.evaluate(world)
        assert len(received) > 0

    def test_emergency_count(self, event_bus):
        engine = SafetyEngine(event_bus=event_bus)
        world = make_world([make_entity(cls="car", distance="near")])
        engine.evaluate(world)
        assert engine.emergency_count >= 1


# ═════════════════════════════════════════════════════════════════════
# SystemContext Tests
# ═════════════════════════════════════════════════════════════════════


class TestSystemContext:
    def test_default_state(self):
        ctx = SystemContext()
        assert ctx.system_state == SystemState.STARTING
        assert ctx.world is None
        assert ctx.location is None

    def test_update_world(self):
        ctx = SystemContext()
        world = make_world([make_entity()], scene_type="street")
        ctx.update_world(world)
        assert ctx.world is not None
        assert ctx.spatial.scene_type == "street"
        assert ctx.spatial.is_indoors is False

    def test_update_indoor(self):
        ctx = SystemContext()
        world = make_world([], scene_type="indoor")
        ctx.update_world(world)
        assert ctx.spatial.is_indoors is True

    def test_update_location(self):
        from core.models.location import GeoLocation
        ctx = SystemContext()
        loc = GeoLocation(latitude=40.0, longitude=-74.0)
        ctx.update_location(loc)
        assert ctx.location is not None
        assert ctx.location.latitude == pytest.approx(40.0)

    def test_navigation_context(self):
        ctx = SystemContext()
        ctx.update_navigation(
            active=True, destination="Central Park",
            distance_remaining_m=500.0,
        )
        assert ctx.navigation.active is True
        assert ctx.navigation.destination == "Central Park"
        assert ctx.navigation.distance_remaining_m == 500.0

    def test_preferences(self):
        ctx = SystemContext()
        ctx.set_preference("volume", 80)
        assert ctx.get_preference("volume") == 80
        assert ctx.get_preference("nonexistent", "default") == "default"

    def test_summary(self):
        ctx = SystemContext()
        s = ctx.summary()
        assert s["system_state"] == "STARTING"
        assert s["has_world"] is False

    def test_session_duration(self):
        ctx = SystemContext()
        time.sleep(0.1)
        assert ctx.session_duration_s > 0.0


# ═════════════════════════════════════════════════════════════════════
# AnnouncementTracker Tests
# ═════════════════════════════════════════════════════════════════════


class TestAnnouncementTracker:
    def test_not_announced_initially(self):
        tracker = AnnouncementTracker()
        assert not tracker.was_announced("car:left")

    def test_mark_and_check(self):
        tracker = AnnouncementTracker()
        tracker.mark_announced("car:left")
        assert tracker.was_announced("car:left")

    def test_different_key_not_affected(self):
        tracker = AnnouncementTracker()
        tracker.mark_announced("car:left")
        assert not tracker.was_announced("car:right")

    def test_cooldown_expires(self):
        tracker = AnnouncementTracker(default_cooldown_s=0.1)
        tracker.mark_announced("car:left")
        time.sleep(0.2)
        assert not tracker.was_announced("car:left")

    def test_hazard_shorter_cooldown(self):
        tracker = AnnouncementTracker(
            default_cooldown_s=100.0, hazard_cooldown_s=0.1
        )
        tracker.mark_announced("hazard:car:left")
        assert tracker.was_announced("hazard:car:left")
        time.sleep(0.2)
        assert not tracker.was_announced("hazard:car:left")

    def test_face_cooldown(self):
        tracker = AnnouncementTracker(face_cooldown_s=0.1)
        tracker.mark_announced("face:Alice")
        assert tracker.was_announced("face:Alice")
        time.sleep(0.2)
        assert not tracker.was_announced("face:Alice")

    def test_entry_count(self):
        tracker = AnnouncementTracker()
        tracker.mark_announced("a")
        tracker.mark_announced("b")
        assert tracker.entry_count == 2

    def test_clear(self):
        tracker = AnnouncementTracker()
        tracker.mark_announced("a")
        tracker.clear()
        assert not tracker.was_announced("a")

    def test_max_entries_eviction(self):
        tracker = AnnouncementTracker(max_entries=5)
        for i in range(10):
            tracker.mark_announced(f"key_{i}")
        assert tracker.entry_count <= 5


# ═════════════════════════════════════════════════════════════════════
# SessionMemory Tests
# ═════════════════════════════════════════════════════════════════════


class TestSessionMemory:
    def test_store_and_recall(self):
        mem = SessionMemory()
        mem.store("car_count", 5)
        assert mem.recall("car_count") == 5

    def test_recall_missing_returns_none(self):
        mem = SessionMemory()
        assert mem.recall("nonexistent") is None

    def test_ttl_expiration(self):
        mem = SessionMemory()
        mem.store("temp", "value", ttl_s=0.1)
        assert mem.recall("temp") == "value"
        time.sleep(0.2)
        assert mem.recall("temp") is None

    def test_recall_by_category(self):
        mem = SessionMemory()
        mem.store("a", 1, category="detection")
        mem.store("b", 2, category="detection")
        mem.store("c", 3, category="face")
        results = mem.recall_by_category("detection")
        assert len(results) == 2

    def test_increment(self):
        mem = SessionMemory()
        assert mem.increment("count") == 1
        assert mem.increment("count") == 2
        assert mem.increment("count", 3) == 5

    def test_forget(self):
        mem = SessionMemory()
        mem.store("key", "value")
        assert mem.forget("key")
        assert mem.recall("key") is None
        assert not mem.forget("nonexistent")

    def test_clear(self):
        mem = SessionMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        mem.clear()
        assert mem.entry_count == 0

    def test_max_entries_eviction(self):
        mem = SessionMemory(max_entries=5)
        for i in range(10):
            mem.store(f"key_{i}", i)
        assert mem.entry_count <= 5


# ═════════════════════════════════════════════════════════════════════
# PersistentMemory Tests
# ═════════════════════════════════════════════════════════════════════


class TestPersistentMemory:
    def test_store_and_recall(self, tmp_path):
        mem = PersistentMemory(filepath=str(tmp_path / "mem.json"))
        mem.store("name", "SIMON")
        assert mem.recall("name") == "SIMON"

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem1 = PersistentMemory(filepath=path)
        mem1.store("key", "value")
        # Create a new instance (simulates restart)
        mem2 = PersistentMemory(filepath=path)
        assert mem2.recall("key") == "value"

    def test_forget(self, tmp_path):
        mem = PersistentMemory(filepath=str(tmp_path / "mem.json"))
        mem.store("key", "value")
        assert mem.forget("key")
        assert mem.recall("key") is None

    def test_list_keys(self, tmp_path):
        mem = PersistentMemory(filepath=str(tmp_path / "mem.json"))
        mem.store("a", 1)
        mem.store("b", 2)
        assert set(mem.list_keys()) == {"a", "b"}

    def test_clear(self, tmp_path):
        mem = PersistentMemory(filepath=str(tmp_path / "mem.json"))
        mem.store("a", 1)
        mem.clear()
        assert mem.list_keys() == []


# ═════════════════════════════════════════════════════════════════════
# MemoryEntry Tests
# ═════════════════════════════════════════════════════════════════════


class TestMemoryEntry:
    def test_not_expired_without_ttl(self):
        entry = MemoryEntry(key="k", value="v")
        assert not entry.is_expired

    def test_expired_with_ttl(self):
        entry = MemoryEntry(key="k", value="v", ttl_s=0.1)
        time.sleep(0.2)
        assert entry.is_expired
