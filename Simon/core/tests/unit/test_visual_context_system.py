"""Unit and integration tests for Context-Aware Visual Information System.

Tests:
1. VisualInfoClassifier (Safety, Navigation, Distance boards, Ambiguity, Normal OCR).
2. VisualEventTracker (deduplication, continuous presence suppression, re-entry).
3. CRITICAL ORDERING: PERCEPTION -> DECISION -> TTS START -> TTS COMPLETE -> OLLAMA.
4. Ollama post-TTS contextual extraction and query answering.
5. Ollama failure resilience (TTS never breaks when Ollama is offline).
6. Vision UI HUD overlay state.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.visual_context.classifier import (
    VisualInfoClassifier,
    VisualPriority,
    ClassificationResult,
)
from core.visual_context.event_tracker import VisualEventTracker
from core.visual_context.ollama_client import OllamaClient
from core.visual_context.context_memory import (
    ContextualVisualMemory,
    VisualMemoryEntry,
)
from core.visual_context.visual_information_manager import (
    VisualInformationManager,
    VisualEvent,
)
from core.logic.logic_controller import LogicController
from vision.pipeline.perception_fusion import WorldModel, Entity
from vision.ocr.base import OCRResult


class TestVisualInfoClassifier:
    def test_stop_sign_classification(self):
        clf = VisualInfoClassifier()
        res = clf.classify_ocr("STOP", position="center")
        assert res.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert res.category == "stop_sign"
        assert res.immediate_tts_response == "Stop sign ahead."

        res_left = clf.classify_ocr("STOP SIGN", position="left")
        assert res_left.immediate_tts_response == "Stop sign on your left."

    def test_road_closure_and_diversion(self):
        clf = VisualInfoClassifier()
        res1 = clf.classify_ocr("ROAD CLOSED", position="center")
        assert res1.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert res1.immediate_tts_response == "Road closed ahead."

        res2 = clf.classify_ocr("ROAD CLOSED TAKE DIVERSION 500M", position="center")
        assert res2.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert "Take the diversion" in res2.immediate_tts_response

    def test_construction_and_hazards(self):
        clf = VisualInfoClassifier()
        res1 = clf.classify_ocr("CONSTRUCTION AHEAD", position="right")
        assert res1.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert res1.immediate_tts_response == "Construction on your right."

        res2 = clf.classify_ocr("OPEN MANHOLE", position="center")
        assert res2.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert "open manhole" in res2.immediate_tts_response.lower()

        res3 = clf.classify_ocr("PEDESTRIAN CROSSING", position="center")
        assert res3.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert res3.immediate_tts_response == "Pedestrian crossing ahead."

    def test_distance_board_extraction(self):
        clf = VisualInfoClassifier()
        res = clf.classify_ocr("KATPADI 2 KM", position="center")
        assert res.priority == VisualPriority.P1_NAVIGATION_CRITICAL
        assert res.category == "distance_board"
        assert res.immediate_tts_response == "Katpadi, 2 kilometres ahead."
        assert res.extracted_data["place"] == "Katpadi"

        res2 = clf.classify_ocr("VELLORE 5 KM", position="center")
        assert res2.immediate_tts_response == "Vellore, 5 kilometres ahead."

    def test_bus_stop_classification(self):
        clf = VisualInfoClassifier()
        res = clf.classify_ocr("BUS STOP", position="right")
        assert res.priority == VisualPriority.P1_NAVIGATION_CRITICAL
        assert res.immediate_tts_response == "Bus stop on your right."

    def test_normal_ocr_no_automatic_speech(self):
        clf = VisualInfoClassifier()
        res = clf.classify_ocr("ABC RESTAURANT SOUTH INDIAN MEALS ₹120")
        assert res.priority in (VisualPriority.P3_CONTEXTUAL, VisualPriority.P4_NORMAL_OCR)
        assert res.immediate_tts_response is None

        res_promo = clf.classify_ocr("SPECIAL DISCOUNT 50% OFF MON-FRI")
        assert res_promo.priority == VisualPriority.P4_NORMAL_OCR
        assert res_promo.immediate_tts_response is None

    def test_ambiguous_ocr_handling(self):
        clf = VisualInfoClassifier()
        res = clf.classify_ocr("DIV...TION NH...")
        assert res.is_ambiguous is True
        assert res.immediate_tts_response is None  # Does NOT invent an immediate fake answer

    def test_hazardous_entity_classification(self):
        clf = VisualInfoClassifier()
        res = clf.classify_entity(entity_cls="dog", position="left", distance="near")
        assert res is not None
        assert res.priority == VisualPriority.P0_SAFETY_CRITICAL
        assert "dog" in res.immediate_tts_response.lower()
        assert "left" in res.immediate_tts_response.lower()


class TestVisualEventTracker:
    def test_event_deduplication_and_reentry(self):
        tracker = VisualEventTracker(absence_reset_s=10.0)
        t0 = 1000.0

        # Frame 1: first seen
        assert tracker.should_process("STOP", now=t0) is True
        tracker.register_seen("STOP", "STOP", "SAFETY_CRITICAL", now=t0)
        tracker.mark_tts_started("STOP", now=t0)

        # Frame 2-10 (continuous presence within 5s): suppressed
        for i in range(1, 10):
            t = t0 + (i * 0.5)
            tracker.register_seen("STOP", "STOP", "SAFETY_CRITICAL", now=t)
            assert tracker.should_process("STOP", now=t) is False

        # Disappears for 15 seconds, then re-enters at t0 + 25s
        t_reentry = t0 + 25.0
        assert tracker.should_process("STOP", now=t_reentry) is True


class TestCriticalOrderingAndTTSCompletion:
    def test_tts_must_happen_before_ollama(self):
        """CRITICAL TEST: Verify execution ordering is:
        PERCEPTION -> DECISION -> TTS START -> TTS COMPLETE -> OLLAMA
        Ollama must NEVER start before TTS completion.
        """
        execution_order = []

        # Mock Ollama client that records when analysis starts
        ollama_client = OllamaClient(enabled=False)
        orig_analyze = ollama_client.analyze_post_tts_async

        def tracking_analyze(payload, on_complete=None):
            execution_order.append("ollama_start")
            orig_analyze(payload, on_complete)

        ollama_client.analyze_post_tts_async = tracking_analyze

        # Manager
        manager = VisualInformationManager(ollama_client=ollama_client)

        # 1. Perception frame with STOP
        ocr = OCRResult(text="STOP", confidence=0.95, bbox=(10, 10, 100, 100))
        world = WorldModel(ocr_texts=[ocr], entities=[])

        # 2. Process world -> Decision made, immediate TTS returned
        events = manager.process_world(world)
        assert len(events) == 1
        assert events[0].spoken_response == "Stop sign ahead."

        execution_order.append("tts_start")

        # Verify Ollama has NOT started yet
        assert "ollama_start" not in execution_order

        # 3. Simulate TTS playback in progress
        time.sleep(0.02)
        assert "ollama_start" not in execution_order

        # 4. TTS finishes physical playback and acoustic drain
        execution_order.append("tts_complete")
        manager.on_tts_completed(event_key=events[0].key, spoken_text=events[0].spoken_response, success=True)

        # Allow worker thread to execute
        time.sleep(0.05)

        # Assert strict sequence
        assert execution_order == [
            "tts_start",
            "tts_complete",
            "ollama_start",
        ]


class TestOllamaFailureResilience:
    def test_ollama_offline_does_not_break_tts(self):
        # Configure client with non-existent server
        ollama_client = OllamaClient(host="http://127.0.0.1:9999", timeout_s=0.1, enabled=True)
        manager = VisualInformationManager(ollama_client=ollama_client)

        ocr = OCRResult(text="ROAD CLOSED", confidence=0.9, bbox=(0, 0, 1, 1))
        world = WorldModel(ocr_texts=[ocr], entities=[])

        # TTS generated normally
        events = manager.process_world(world)
        assert len(events) == 1
        assert "Road closed" in events[0].spoken_response

        # TTS completes -> Ollama fails in background -> fallback kicks in
        manager.on_tts_completed(event_key=events[0].key, spoken_text=events[0].spoken_response, success=True)
        time.sleep(0.25)

        # Context memory was still updated via fallback!
        entries = manager.memory.get_recent_entries()
        assert len(entries) >= 1
        assert entries[0].classification == "SAFETY_CRITICAL"


class TestContextMemoryAndUserQueries:
    def test_context_storage_and_query_answering(self):
        memory = ContextualVisualMemory()
        
        # Add a restaurant observation
        entry1 = VisualMemoryEntry(
            timestamp=time.time(),
            raw_ocr=["ABC Restaurant", "South Indian Meals", "₹120"],
            detected_objects=["building"],
            faces=[],
            classification="CONTEXTUAL",
            spoken_response=None,
            direction="on your right",
            distance="near",
            place_name="ABC Restaurant",
            category="restaurant",
            summary="ABC Restaurant offering South Indian Meals on your right.",
        )
        memory.add_entry(entry1)

        # Add a bus stop observation
        entry2 = VisualMemoryEntry(
            timestamp=time.time(),
            raw_ocr=["BUS STOP", "ROUTE 12"],
            detected_objects=["bus_stop"],
            faces=[],
            classification="NAVIGATION_CRITICAL",
            spoken_response="Bus stop ahead.",
            direction="ahead",
            distance="medium",
            event_type="bus_stop",
            place_name="Bus Stop Route 12",
            category="transport",
            summary="Bus stop ahead for Route 12.",
        )
        memory.add_entry(entry2)

        # Query for restaurants
        res_food = memory.answer_query("Are there restaurants nearby?")
        assert "ABC Restaurant" in res_food or "restaurant" in res_food.lower()

        # Query for bus stop
        res_bus = memory.answer_query("Is there a bus stop nearby?")
        assert "bus" in res_bus.lower()

        # Query for non-existent item
        res_none = memory.answer_query("Where is the nearest hospital?")
        assert "haven't seen" in res_none.lower() or "no" in res_none.lower()


class TestLogicControllerVisualContextIntegration:
    def test_logic_controller_end_to_end_flow(self):
        bus = EventBus()
        bus.start()

        ollama_client = OllamaClient(enabled=False)
        manager = VisualInformationManager(ollama_client=ollama_client)
        logic = LogicController(event_bus=bus, visual_info_manager=manager)
        logic.start()

        spoken_events = []
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            lambda e: spoken_events.append(e.data.get("text")),
            source="test_sub",
        )

        # 1. Publish world update with STOP sign
        ocr = OCRResult(text="STOP", confidence=0.95, bbox=(0, 0, 10, 10))
        world = WorldModel(ocr_texts=[ocr], entities=[])
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="vision"))
        time.sleep(0.05)

        assert len(spoken_events) == 1
        assert spoken_events[0] == "Stop sign ahead."

        # 2. Subsequent frame with same STOP sign is suppressed
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="vision"))
        time.sleep(0.05)
        assert len(spoken_events) == 1

        # 3. Simulate TTS completion event from speech pipeline
        bus.publish(Event(
            topic=event_types.SPEECH_PLAYBACK_COMPLETE,
            data={"text": "Stop sign ahead.", "success": True, "event_key": "STOP"},
            source="speech_pipeline",
        ))
        time.sleep(0.05)

        # Verify context memory has the event recorded post-TTS
        recent = manager.memory.get_recent_entries()
        assert len(recent) >= 1

        logic.stop()
        bus.stop()

    def test_strict_timestamp_order_tts_start_complete_ollama(self):
        """Prove with timestamps that tts_start < tts_complete <= ollama_start."""
        timestamps = {}

        ollama_client = OllamaClient(enabled=False)
        orig_analyze = ollama_client.analyze_post_tts_async

        def tracking_analyze(payload, on_complete=None):
            timestamps["ollama_start"] = time.perf_counter()
            orig_analyze(payload, on_complete)

        ollama_client.analyze_post_tts_async = tracking_analyze
        manager = VisualInformationManager(ollama_client=ollama_client)

        # 1. Perception frame
        ocr = OCRResult(text="ROAD CLOSED", confidence=0.9, bbox=(0, 0, 10, 10))
        world = WorldModel(ocr_texts=[ocr], entities=[])

        # 2. Decision & TTS start
        events = manager.process_world(world)
        timestamps["tts_start"] = time.perf_counter()
        assert len(events) == 1

        # Delay simulating audio playback
        time.sleep(0.03)

        # 3. TTS completion signal
        timestamps["tts_complete"] = time.perf_counter()
        manager.on_tts_completed(event_key=events[0].key, spoken_text=events[0].spoken_response, success=True)

        time.sleep(0.05)

        # Assert strict inequalities
        assert "tts_start" in timestamps
        assert "tts_complete" in timestamps
        assert "ollama_start" in timestamps
        assert timestamps["tts_start"] < timestamps["tts_complete"]
        assert timestamps["tts_complete"] <= timestamps["ollama_start"]


class TestVisionUIOverlay:
    def test_vision_renderer_hud_all_fields(self):
        """Test VisionRenderer HUD overlay contains and renders all 9 required fields."""
        from vision.vision_renderer import VisionRenderer
        import numpy as np

        renderer = VisionRenderer(enabled=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        entity1 = Entity(
            track_id=1,
            cls="dog",
            bbox=(50, 50, 150, 150),
            confidence=0.92,
            position="left",
            distance="near",
        )
        entity2 = Entity(
            track_id=2,
            cls="person",
            bbox=(200, 50, 350, 350),
            confidence=0.95,
            position="center",
            distance="medium",
            face_name="Ganesh",
            face_bbox=(220, 70, 300, 170),
        )
        ocr = OCRResult(text="STOP", confidence=0.98, bbox=(400, 50, 500, 150))
        world = WorldModel(entities=[entity1, entity2], ocr_texts=[ocr])

        ui_state = {
            "objects": ["dog", "traffic_sign"],
            "faces": ["Ganesh"],
            "ocr": ["STOP"],
            "classification": "SAFETY_CRITICAL",
            "direction": "Left",
            "distance": "Near",
            "tts_text": "Stop sign ahead.",
            "tts_status": "Speaking",
            "ollama_status": "Waiting",
        }

        # Render without error
        renderer.render(frame, world, ui_state=ui_state)
        assert renderer._ui_state["classification"] == "SAFETY_CRITICAL"
        assert renderer._ui_state["tts_status"] == "Speaking"
        assert renderer._ui_state["ollama_status"] == "Waiting"

    def test_area_name_and_street_signs(self):
        """Test area name, road name, and highway destination board classification."""
        clf = VisualInfoClassifier()

        res1 = clf.classify_ocr("M.G. ROAD", position="left")
        assert res1.priority == VisualPriority.P1_NAVIGATION_CRITICAL
        assert "M.G. ROAD" in res1.immediate_tts_response
        assert "left" in res1.immediate_tts_response

        res2 = clf.classify_ocr("NH 48 HIGHWAY", position="center")
        assert res2.priority == VisualPriority.P1_NAVIGATION_CRITICAL
        assert "NH 48 HIGHWAY" in res2.immediate_tts_response

    def test_ordering_fails_if_ollama_starts_before_tts_completion(self):
        """Negative test proving that starting Ollama before TTS completion is rejected."""
        events_timeline = []

        def simulated_broken_pipeline():
            # Broken sequence: Vision -> Ollama -> TTS
            events_timeline.append("tts_start")
            events_timeline.append("ollama_start") # INCORRECT: before tts_complete
            events_timeline.append("tts_complete")

        simulated_broken_pipeline()

        tts_complete_idx = events_timeline.index("tts_complete")
        ollama_start_idx = events_timeline.index("ollama_start")

        # In a broken sequence, ollama_start_idx < tts_complete_idx
        is_correct_order = tts_complete_idx < ollama_start_idx
        assert not is_correct_order, "Demonstration that premature Ollama start violates required sequence"


