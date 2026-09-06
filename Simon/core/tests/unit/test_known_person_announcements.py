"""Unit and integration tests for Known Person Arrival Announcement System.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from core.events.event_bus import EventBus
from core.models.events import Event
from core.events import event_types
from core.models.enums import Priority
from core.logic.known_person_tracker import KnownPersonTracker, KNOWN_PERSON_REANNOUNCE_TIMEOUT_S

from core.logic.language_generator import LanguageGenerator
from core.logic.logic_controller import LogicController
from vision.pipeline.perception_fusion import WorldModel, Entity


class TestKnownPersonTracker:
    def test_first_detection_announces_and_records_state(self):
        tracker = KnownPersonTracker()
        t0 = 1000.0
        announce = tracker.process_detection("Ganesh", now=t0)
        assert announce is True
        state = tracker.get_state("Ganesh")
        assert state is not None
        assert state.name == "Ganesh"
        assert state.first_seen == t0
        assert state.last_seen == t0
        assert state.last_announced == t0
        assert state.announcement_count == 1


    def test_same_person_across_many_frames_announces_only_once(self):
        tracker = KnownPersonTracker()
        t0 = 1000.0
        assert tracker.process_detection("Ganesh", now=t0) is True

        # 100 subsequent frames within 30 seconds
        for i in range(1, 101):
            frame_t = t0 + (i * 0.3)
            assert tracker.process_detection("Ganesh", now=frame_t) is False


    def test_temporary_disappearance_under_5_minutes_suppresses_announcement(self):
        tracker = KnownPersonTracker()
        t0 = 1000.0
        assert tracker.process_detection("Ganesh", now=t0) is True

        # Disappears for 42.3 seconds and returns
        assert tracker.process_detection("Ganesh", now=t0 + 42.3) is False

        # Disappears for 299.0 seconds and returns
        assert tracker.process_detection("Ganesh", now=t0 + 42.3 + 299.0) is False


    def test_reentry_after_5_minutes_announces(self):
        tracker = KnownPersonTracker()
        t0 = 1000.0
        assert tracker.process_detection("Ganesh", now=t0) is True

        # Disappears for 301.2 seconds (> 5 minutes)
        assert tracker.process_detection("Ganesh", now=t0 + 301.2) is True
        state = tracker.get_state("Ganesh")
        assert state.announcement_count == 2
        assert state.last_seen == t0 + 301.2

        # Subsequent frame after re-entry does not reannounce
        assert tracker.process_detection("Ganesh", now=t0 + 301.5 + 0.5) is False


    def test_multiple_people_tracked_independently(self):
        tracker = KnownPersonTracker()
        t0 = 1000.0

        # Ganesh detected -> announce
        assert tracker.process_detection("Ganesh", now=t0) is True

        # Ganesh remains -> no repeat
        assert tracker.process_detection("Ganesh", now=t0 + 1.0) is False

        # Lokesh detected -> announce
        assert tracker.process_detection("Lokesh", now=t0 + 2.0) is True

        # Ganesh remains -> no repeat
        assert tracker.process_detection("Ganesh", now=t0 + 3.0) is False

        # Lokesh remains -> no repeat
        assert tracker.process_detection("Lokesh", now=t0 + 4.0) is False

        # Both disappear. Ganesh returns after 6 minutes
        assert tracker.process_detection("Ganesh", now=t0 + 365.0) is True

        # Lokesh returns after 1 minute (before 5 min re-entry)
        assert tracker.process_detection("Lokesh", now=t0 + 65.0) is False


    def test_unknown_faces_ignored(self):
        tracker = KnownPersonTracker()
        assert tracker.process_detection("Unknown") is False
        assert tracker.process_detection("") is False
        assert tracker.process_detection(None) is False


    def test_custom_reannounce_timeout(self):
        tracker = KnownPersonTracker(reannounce_timeout_s=60.0)
        t0 = 1000.0
        assert tracker.process_detection("Ganesh", now=t0) is True
        assert tracker.process_detection("Ganesh", now=t0 + 50.0) is False
        assert tracker.process_detection("Ganesh", now=t0 + 50.0 + 65.0) is True



class TestSpatialArrivalFormatting:
    def test_known_person_on_left(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="left") == "Ganesh is here, on your left."

    def test_known_person_on_right(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="right") == "Ganesh is here, on your right."

    def test_known_person_directly_ahead(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="center") == "Ganesh is here, directly ahead."
        assert gen.known_person_arrival("Ganesh", position="ahead") == "Ganesh is here, directly ahead."

    def test_known_person_slightly_left_and_right(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="slightly_left") == "Ganesh is here, slightly on your left."
        assert gen.known_person_arrival("Ganesh", position="slightly_right") == "Ganesh is here, slightly on your right."

    def test_known_person_nearby_on_left_right(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="left", distance="near") == "Ganesh is here, nearby on your left."
        assert gen.known_person_arrival("Ganesh", position="right", distance="near") == "Ganesh is here, nearby on your right."

    def test_known_person_far_ahead(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="center", distance="far") == "Ganesh is here, far ahead."
        assert gen.known_person_arrival("Ganesh", position="ahead", distance="far") == "Ganesh is here, far ahead."

    def test_known_person_missing_distance_or_direction(self):
        gen = LanguageGenerator()
        assert gen.known_person_arrival("Ganesh", position="left", distance=None) == "Ganesh is here, on your left."
        assert gen.known_person_arrival("Ganesh", position="left", distance="medium") == "Ganesh is here, on your left."
        assert gen.known_person_arrival("Ganesh", position=None, distance="near") == "Ganesh is here, nearby."
        assert gen.known_person_arrival("Ganesh", position=None, distance="far") == "Ganesh is here, far ahead."
        assert gen.known_person_arrival("Ganesh", position=None, distance=None) == "Ganesh is here."


class TestLogicControllerKnownPersonIntegration:
    def test_logic_controller_announces_known_person_once_with_spatial(self):
        bus = EventBus()
        bus.start()

        tracker = KnownPersonTracker()
        logic = LogicController(event_bus=bus, known_person_tracker=tracker)
        logic.start()

        announcements = []
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            lambda e: announcements.append(e.data.get("text")),
            source="test_subscriber",
        )

        # 1. First detection of Ganesh on left
        entity = Entity(
            track_id=1,
            cls="person",
            bbox=(10, 10, 100, 100),
            confidence=0.95,
            face_name="Ganesh",
            position="left",
            distance="near",
        )
        world = WorldModel(entities=[entity], frame_id=1)

        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="vision"))
        time.sleep(0.05)

        assert len(announcements) == 1
        assert announcements[0] == "Ganesh is here, nearby on your left."

        # 2. Subsequent frame with same Ganesh does not re-announce
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="vision"))
        time.sleep(0.05)

        assert len(announcements) == 1

        # 3. Track ID changes to 7 (e.g. camera tracker jitter), but name is still Ganesh
        entity2_track7 = Entity(
            track_id=7,
            cls="person",
            bbox=(10, 10, 100, 100),
            confidence=0.95,
            face_name="Ganesh",
            position="right",
        )
        world_track7 = WorldModel(entities=[entity2_track7], frame_id=2)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world_track7}, source="vision"))
        time.sleep(0.05)

        assert len(announcements) == 1

        logic.stop()
        bus.stop()


    def test_logic_controller_reentry_after_5_minutes_announces(self):
        bus = EventBus()
        bus.start()

        tracker = KnownPersonTracker(reannounce_timeout_s=300.0)
        logic = LogicController(event_bus=bus, known_person_tracker=tracker)
        logic.start()

        announcements = []
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            lambda e: announcements.append(e.data.get("text")),
            source="test_subscriber",
        )

        t0 = 1000.0
        entity = Entity(track_id=1, cls="person", bbox=(0, 0, 1, 1), confidence=0.9, face_name="Ganesh", position="left")
        world = WorldModel(entities=[entity])

        # First detection at t0
        tracker.clear()
        tracker.process_detection("Ganesh", now=time.time() - 350.0)

        # Re-entry after 350s absence (> 300s timeout) spotted on right
        entity_reentry = Entity(track_id=7, cls="person", bbox=(0, 0, 1, 1), confidence=0.9, face_name="Ganesh", position="right")
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": WorldModel(entities=[entity_reentry])}, source="vision"))
        time.sleep(0.05)

        assert len(announcements) >= 1
        assert announcements[-1] == "Ganesh is here, on your right."

        logic.stop()
        bus.stop()

    def test_unknown_faces_produce_no_known_person_announcement(self):
        bus = EventBus()
        bus.start()

        tracker = KnownPersonTracker()
        logic = LogicController(event_bus=bus, known_person_tracker=tracker)
        logic.start()

        announcements = []
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            lambda e: announcements.append(e.data.get("text")),
            source="test_subscriber",
        )

        entity = Entity(track_id=1, cls="person", bbox=(0, 0, 1, 1), confidence=0.9, face_name="Unknown", position="left")
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": WorldModel(entities=[entity])}, source="vision"))
        time.sleep(0.05)

        # No "is here" announcement made
        assert not any("is here" in a for a in announcements)

        logic.stop()
        bus.stop()
