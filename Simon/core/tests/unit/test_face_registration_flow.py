"""Test suite for SIMON Face Registration & Speech Interaction Scenarios."""

import os
import queue
import shutil
import tempfile
import time
import numpy as np
import pytest

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.logic.face_registration_tracker import FaceRegistrationTracker, TrackerState
from core.logic.logic_controller import LogicController
from core.logic.priority_engine import PriorityEngine
from vision.pipeline.perception_fusion import WorldModel, Entity
from vision.face.insightface_recognizer import InsightFaceRecognizer
from vision.face.base import FaceResult
from core.config.system_config import FaceConfig
from speech.manager.listen_pipeline import ListenPipeline
from speech.config.speech_config import SpeechConfig
from speech.models.speech_command import SpeechCommand
from speech.stt.transcript import Transcript
from speech.stt.confidence_gate import ConfidenceGate
from speech.postprocessing.hallucination_filter import HallucinationFilter
from speech.wakeword.text_match_detector import TextMatchDetector


@pytest.fixture
def temp_face_db():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestFaceRegistrationTracker:
    def test_unknown_face_timing_and_prompt(self, temp_face_db):
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.2,
            grace_period_s=0.5,
            name_timeout_s=0.3,
            face_db_dir=temp_face_db,
        )
        dummy_emb = np.random.randn(512).astype(np.float32)
        
        # Frame 1 at t=0
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world1 = WorldModel(entities=[entity], frame_id=1)
        res = tracker.update_world(world1)
        assert res is None  # < 0.2s, no prompt
        assert tracker.get_active_prompt_state() is None

        # Wait past threshold
        time.sleep(0.25)
        world2 = WorldModel(entities=[entity], frame_id=2)
        res = tracker.update_world(world2)
        assert res == 1  # Threshold met, prompted
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE
        assert tracker.get_active_track_id() == 1
        assert np.array_equal(tracker.get_active_embedding(), dummy_emb)

        # Subsequent frames do not re-trigger
        world3 = WorldModel(entities=[entity], frame_id=3)
        res = tracker.update_world(world3)
        assert res is None

    def test_prompting_save_does_not_expire_on_face_absence(self, temp_face_db):
        """Encounter in PROMPTING_SAVE must not expire simply because face is absent for grace_period."""
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.05,
            grace_period_s=0.1,  # Short grace
            name_timeout_s=0.5,
            face_db_dir=temp_face_db,
        )
        dummy_emb = np.random.randn(512).astype(np.float32)
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        tracker.update_world(world)
        time.sleep(0.06)
        tracker.update_world(world)
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE

        # Face disappears for 0.15s (> 0.1s grace)
        time.sleep(0.15)
        empty_world = WorldModel(entities=[], frame_id=2)
        tracker.update_world(empty_world)

        # Must still be in PROMPTING_SAVE with embedding preserved!
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE
        assert tracker.get_active_track_id() == 1
        assert np.array_equal(tracker.get_active_embedding(), dummy_emb)

    def test_prompting_name_does_not_expire_on_face_absence(self, temp_face_db):
        """Encounter in PROMPTING_NAME must remain alive until named or check_name_timeout() fires."""
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.05,
            grace_period_s=0.1,
            name_timeout_s=0.5,
            face_db_dir=temp_face_db,
        )
        dummy_emb = np.random.randn(512).astype(np.float32)
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        tracker.update_world(world)
        time.sleep(0.06)
        tracker.update_world(world)
        tracker.advance_to_name_prompt()
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_NAME

        # Disappear for 0.15s
        time.sleep(0.15)
        empty_world = WorldModel(entities=[], frame_id=2)
        tracker.update_world(empty_world)

        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_NAME
        assert tracker.get_active_track_id() == 1
        assert np.array_equal(tracker.get_active_embedding(), dummy_emb)

    def test_user_declined_state(self, temp_face_db):
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.1, face_db_dir=temp_face_db)
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown")
        world = WorldModel(entities=[entity], frame_id=1)
        
        tracker.update_world(world)
        time.sleep(0.15)
        tracker.update_world(world)
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE

        tracker.mark_declined()
        assert tracker.get_active_prompt_state() is None

        # Same encounter should not prompt again
        time.sleep(0.1)
        assert tracker.update_world(world) is None

    def test_multiple_unknown_faces_independence(self, temp_face_db):
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.1, face_db_dir=temp_face_db)
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])
        e1 = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=emb1)
        e2 = Entity(track_id=2, cls="person", bbox=(60, 10, 100, 50), confidence=0.9, face_name="Unknown", face_embedding=emb2)
        
        world = WorldModel(entities=[e1, e2], frame_id=1)
        tracker.update_world(world)
        time.sleep(0.15)
        res = tracker.update_world(world)
        assert res == 1  # First person gets prompted
        assert np.array_equal(tracker.get_active_embedding(), emb1)

        # e2 is tracking independently and does not overwrite e1's encounter
        enc1 = tracker._encounters[1]
        enc2 = tracker._encounters[2]
        assert np.array_equal(enc1.embedding, emb1)
        assert np.array_equal(enc2.embedding, emb2)

    def test_name_timeout(self, temp_face_db):
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, name_timeout_s=0.1, face_db_dir=temp_face_db)
        e = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=np.zeros(10))
        world = WorldModel(entities=[e], frame_id=1)
        tracker.update_world(world)
        time.sleep(0.06)
        tracker.update_world(world)
        tracker.advance_to_name_prompt()

        assert not tracker.check_name_timeout()
        time.sleep(0.12)
        assert tracker.check_name_timeout()


class TestLogicControllerRegistrationFlow:
    def test_end_to_end_named_registration(self, temp_face_db):
        bus = EventBus()
        bus.start()
        
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, face_db_dir=temp_face_db)
        controller = LogicController(event_bus=bus)
        controller._face_tracker = tracker
        controller.start()

        announcements = []
        saved_faces = []
        bus.subscribe(event_types.LOGIC_ANNOUNCE, lambda e: announcements.append(e.data.get("text")), source="test")
        bus.subscribe(event_types.VISION_SAVE_FACE, lambda e: saved_faces.append(e.data), source="test")

        dummy_emb = np.array([0.5, 0.5, 0.5])
        entity = Entity(track_id=42, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        # 1. World update triggers tracking
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.08)
        # 2. Threshold passed -> question asked
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.05)

        assert "Would you like me to save this person?" in announcements

        # 3. User says "yes"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "yes", "args": "", "raw_transcript": "yes"}, source="test"))
        time.sleep(0.05)

        assert "What should I call this person?" in announcements

        # 4. User says "Alice"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "unmatched_text", "args": "Alice", "raw_transcript": "Alice"}, source="test"))
        time.sleep(0.05)

        assert "Alice saved." in announcements
        assert len(saved_faces) == 1
        assert saved_faces[0]["name"] == "Alice"
        assert np.array_equal(saved_faces[0]["embedding"], dummy_emb)
        assert tracker.get_active_prompt_state() is None

        controller.stop()
        bus.stop()

    def test_end_to_end_anonymous_registration_via_no_name(self, temp_face_db):
        bus = EventBus()
        bus.start()
        
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, face_db_dir=temp_face_db)
        controller = LogicController(event_bus=bus)
        controller._face_tracker = tracker
        controller.start()

        announcements = []
        saved_faces = []
        bus.subscribe(event_types.LOGIC_ANNOUNCE, lambda e: announcements.append(e.data.get("text")), source="test")
        bus.subscribe(event_types.VISION_SAVE_FACE, lambda e: saved_faces.append(e.data), source="test")

        dummy_emb = np.array([0.1, 0.2, 0.3])
        entity = Entity(track_id=99, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.08)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.05)

        # Say yes
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "yes", "args": "", "raw_transcript": "yes"}, source="test"))
        time.sleep(0.05)

        # Say "no name"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "no_name", "args": "", "raw_transcript": "no name"}, source="test"))
        time.sleep(0.05)

        assert len(saved_faces) == 1
        assert saved_faces[0]["name"] == "Person 1"
        assert "Person 1 saved." in announcements

        controller.stop()
        bus.stop()

    def test_end_to_end_decline_saving(self, temp_face_db):
        bus = EventBus()
        bus.start()
        
        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, face_db_dir=temp_face_db)
        controller = LogicController(event_bus=bus)
        controller._face_tracker = tracker
        controller.start()

        announcements = []
        saved_faces = []
        bus.subscribe(event_types.LOGIC_ANNOUNCE, lambda e: announcements.append(e.data.get("text")), source="test")
        bus.subscribe(event_types.VISION_SAVE_FACE, lambda e: saved_faces.append(e.data), source="test")

        dummy_emb = np.array([0.1, 0.2, 0.3])
        entity = Entity(track_id=88, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.08)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.05)

        # Say "no" to saving
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "no", "args": "", "raw_transcript": "no"}, source="test"))
        time.sleep(0.05)

        assert len(saved_faces) == 0
        assert "What should I call this person?" not in announcements
        assert tracker._encounters[88].state == TrackerState.DECLINED

        controller.stop()
        bus.stop()


class TestInsightFaceDatabaseAndRecognition:
    def test_save_and_reload_named_and_anonymous(self, temp_face_db):
        cfg = FaceConfig(database_dir=temp_face_db, similarity_threshold=0.5)
        recognizer = InsightFaceRecognizer(config=cfg)
        recognizer._ready = True

        emb_john = np.random.randn(512).astype(np.float32)
        emb_john /= np.linalg.norm(emb_john)
        face_john = FaceResult(name="Unknown", confidence=1.0, bbox=(0, 0, 10, 10), embedding=emb_john)
        assert recognizer.save_face(None, face_john, "John")

        emb_p1 = np.random.randn(512).astype(np.float32)
        emb_p1 /= np.linalg.norm(emb_p1)
        face_p1 = FaceResult(name="Unknown", confidence=1.0, bbox=(0, 0, 10, 10), embedding=emb_p1)
        assert recognizer.save_face(None, face_p1, "Person 1")

        # Test reload in new instance
        rec2 = InsightFaceRecognizer(config=cfg)
        rec2._load_database()
        assert rec2.known_count == 2
        assert "John" in rec2._known_embeddings
        assert "Person 1" in rec2._known_embeddings

        # Match John
        name, score = rec2._match_embedding(emb_john)
        assert name == "John"
        assert score > 0.99

        # Match Person 1
        name, score = rec2._match_embedding(emb_p1)
        assert name == "Person 1"
        assert score > 0.99


class TestSpeechIntentParsingAndConfidenceGate:
    def test_intent_parsing_boundaries_and_punctuation(self):
        cfg = SpeechConfig()
        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=None,
            stt_engine=None,
            confidence_gate=None,
            hallucination_filter=None,
            config=cfg,
        )

        # "no" vs "Noah" / "Nora"
        assert pipeline._parse_intent("no") == ("no", "")
        assert pipeline._parse_intent("No.") == ("no", "")
        assert pipeline._parse_intent("no thanks") == ("no", "")
        assert pipeline._parse_intent("Noah") == ("unmatched_text", "Noah")
        assert pipeline._parse_intent("Nora") == ("unmatched_text", "Nora")

        # "yes" vs "Yesterday"
        assert pipeline._parse_intent("yes") == ("yes", "")
        assert pipeline._parse_intent("Yes.") == ("yes", "")
        assert pipeline._parse_intent("yes!") == ("yes", "")
        assert pipeline._parse_intent("yes, sir") == ("yes", "sir")
        assert pipeline._parse_intent("Yesterday") == ("unmatched_text", "Yesterday")

        # "no name"
        assert pipeline._parse_intent("no name") == ("no_name", "")

    def test_yaml_boolean_command_map_safety(self):
        """Verify command map containing boolean True/False values parses without error."""
        cfg = SpeechConfig()
        cfg.command_map = {
            True: [True, "yeah", "yep"],
            False: [False, "nope", "no thanks"],
            "no_name": ["no name", "without a name"],
        }
        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=None,
            stt_engine=None,
            confidence_gate=None,
            hallucination_filter=None,
            config=cfg,
        )

        assert pipeline._parse_intent("yes") == ("yes", "")
        assert pipeline._parse_intent("yeah") == ("yes", "")
        assert pipeline._parse_intent("no") == ("no", "")
        assert pipeline._parse_intent("nope") == ("no", "")
        assert pipeline._parse_intent("no name") == ("no_name", "")
        assert pipeline._parse_intent("John") == ("unmatched_text", "John")

    def test_yes_transcript_confidence_54_accepted(self):
        """Verify 'Yes' transcript with confidence 0.54 passes through confidence gate as action='yes'."""
        cfg = SpeechConfig()
        cfg.wakeword.always_listen = True
        gate = ConfidenceGate(thresholds={"yes": 0.50, "no": 0.50, "unmatched_text": 0.50})
        
        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=TextMatchDetector("simon"),
            stt_engine=None,
            confidence_gate=gate,
            hallucination_filter=HallucinationFilter(),
            config=cfg,
        )

        # Mock STT engine return
        mock_transcript = Transcript(text="Yes.", raw_text="Yes.", confidence=0.54, engine_name="faster-whisper")
        pipeline._stt = type("MockSTT", (), {"transcribe": lambda self, seg: mock_transcript})()

        pipeline._process_speech_segment(np.zeros(1600, dtype=np.float32))

        # Command queue should have the approved command
        assert not pipeline._command_queue.empty()
        cmd = pipeline._command_queue.get_nowait()
        assert cmd.action == "yes"
        assert cmd.confidence == 0.54
        assert cmd.raw_transcript == "Yes."

    def test_yes_transcript_confidence_below_threshold_blocked(self):
        """Verify 'Yes' transcript with confidence 0.40 is blocked by confidence gate."""
        cfg = SpeechConfig()
        cfg.wakeword.always_listen = True
        gate = ConfidenceGate(thresholds={"yes": 0.50, "no": 0.50})
        
        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=TextMatchDetector("simon"),
            stt_engine=None,
            confidence_gate=gate,
            hallucination_filter=HallucinationFilter(),
            config=cfg,
        )

        mock_transcript = Transcript(text="Yes.", raw_text="Yes.", confidence=0.40, engine_name="faster-whisper")
        pipeline._stt = type("MockSTT", (), {"transcribe": lambda self, seg: mock_transcript})()

        pipeline._process_speech_segment(np.zeros(1600, dtype=np.float32))
        assert pipeline._command_queue.empty()

    def test_spoken_name_confidence_46_accepted_when_expecting_name(self):
        """Verify 'Ganesh' with confidence 0.46 is accepted when expecting_name=True."""
        cfg = SpeechConfig()
        cfg.wakeword.always_listen = True
        gate = ConfidenceGate(thresholds={"yes": 0.50, "no": 0.50, "unmatched_text": 0.50, "name": 0.40}, expecting_name=True)

        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=TextMatchDetector("simon"),
            stt_engine=None,
            confidence_gate=gate,
            hallucination_filter=HallucinationFilter(),
            config=cfg,
        )

        mock_transcript = Transcript(text="Ganesh.", raw_text="Ganesh.", confidence=0.46, engine_name="faster-whisper")
        pipeline._stt = type("MockSTT", (), {"transcribe": lambda self, seg: mock_transcript})()

        pipeline._process_speech_segment(np.zeros(1600, dtype=np.float32))

        assert not pipeline._command_queue.empty()
        cmd = pipeline._command_queue.get_nowait()
        assert cmd.action == "unmatched_text"
        assert cmd.args == "Ganesh"
        assert cmd.confidence == 0.46

    def test_spoken_name_confidence_below_40_rejected(self):
        """Verify 'Ganesh' with confidence 0.30 is rejected even when expecting_name=True."""
        cfg = SpeechConfig()
        cfg.wakeword.always_listen = True
        gate = ConfidenceGate(thresholds={"yes": 0.50, "no": 0.50, "unmatched_text": 0.50, "name": 0.40}, expecting_name=True)

        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=TextMatchDetector("simon"),
            stt_engine=None,
            confidence_gate=gate,
            hallucination_filter=HallucinationFilter(),
            config=cfg,
        )

        mock_transcript = Transcript(text="Ganesh.", raw_text="Ganesh.", confidence=0.30, engine_name="faster-whisper")
        pipeline._stt = type("MockSTT", (), {"transcribe": lambda self, seg: mock_transcript})()

        pipeline._process_speech_segment(np.zeros(1600, dtype=np.float32))
        assert pipeline._command_queue.empty()

    def test_spoken_name_outside_expecting_name_requires_50(self):
        """Verify 'Ganesh' with confidence 0.46 is rejected outside name-prompt context (expecting_name=False)."""
        cfg = SpeechConfig()
        cfg.wakeword.always_listen = True
        gate = ConfidenceGate(thresholds={"yes": 0.50, "no": 0.50, "unmatched_text": 0.50, "name": 0.40}, expecting_name=False)

        pipeline = ListenPipeline(
            audio_stream=None,
            dsp_pipeline=None,
            vad=None,
            wake_word_detector=TextMatchDetector("simon"),
            stt_engine=None,
            confidence_gate=gate,
            hallucination_filter=HallucinationFilter(),
            config=cfg,
        )

        mock_transcript = Transcript(text="Ganesh.", raw_text="Ganesh.", confidence=0.46, engine_name="faster-whisper")
        pipeline._stt = type("MockSTT", (), {"transcribe": lambda self, seg: mock_transcript})()

        pipeline._process_speech_segment(np.zeros(1600, dtype=np.float32))
        assert pipeline._command_queue.empty()

    def test_end_to_end_named_registration_ganesh_saves_actual_name(self, temp_face_db):
        """Verify full flow where 'Ganesh' at confidence 0.46 is saved with actual name and not Person 2."""
        bus = EventBus()
        bus.start()

        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, name_timeout_s=15.0, face_db_dir=temp_face_db)
        controller = LogicController(event_bus=bus)
        controller._face_tracker = tracker
        controller.start()

        announcements = []
        saved_faces = []
        speech_context_events = []
        bus.subscribe(event_types.LOGIC_ANNOUNCE, lambda e: announcements.append(e.data.get("text")), source="test")
        bus.subscribe(event_types.VISION_SAVE_FACE, lambda e: saved_faces.append(e.data), source="test")
        bus.subscribe(event_types.SPEECH_SET_CONTEXT, lambda e: speech_context_events.append(e.data), source="test")

        dummy_emb = np.random.randn(512).astype(np.float32)
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.08)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.05)

        assert "Would you like me to save this person?" in announcements

        # 1. User says "yes"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "yes", "args": "", "raw_transcript": "yes"}, source="test"))
        time.sleep(0.05)

        assert "What should I call this person?" in announcements
        assert any(e.get("expecting_name") is True for e in speech_context_events)

        # 2. User says "Ganesh"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "unmatched_text", "args": "Ganesh", "raw_transcript": "Ganesh."}, source="test"))
        time.sleep(0.05)

        assert len(saved_faces) == 1
        assert saved_faces[0]["name"] == "Ganesh"
        assert saved_faces[0]["name"] != "Person 2"
        assert np.array_equal(saved_faces[0]["embedding"], dummy_emb)
        assert "Ganesh saved." in announcements
        assert tracker.get_active_prompt_state() is None

        controller.stop()
        bus.stop()

    def test_end_to_end_ganesh_saves_to_disk_with_exact_embedding(self, tmp_path):
        """Verify full flow from SpeechCommand to VisionPipeline creating data/faces/Ganesh/0.npy on disk."""
        db_dir = str(tmp_path / "faces")
        bus = EventBus()
        bus.start()

        # Face recognizer config
        from vision.face.insightface_recognizer import InsightFaceRecognizer, FaceConfig
        cfg = FaceConfig(database_dir=db_dir)
        recognizer = InsightFaceRecognizer(cfg)

        # Vision pipeline mock to handle _on_save_face
        from vision.pipeline.vision_pipeline import VisionPipeline
        mock_pipeline = type("MockVisionPipeline", (), {
            "_face_recognizer": recognizer,
            "_on_save_face": VisionPipeline._on_save_face,
        })()
        bus.subscribe(
            event_types.VISION_SAVE_FACE,
            lambda e: VisionPipeline._on_save_face(mock_pipeline, e),
            source="vision_pipeline",
        )

        tracker = FaceRegistrationTracker(visibility_threshold_s=0.05, name_timeout_s=15.0, face_db_dir=db_dir)
        controller = LogicController(event_bus=bus)
        controller._face_tracker = tracker
        controller.start()

        dummy_emb = np.random.randn(512).astype(np.float32)
        entity = Entity(track_id=1, cls="person", bbox=(10, 10, 50, 50), confidence=0.9, face_name="Unknown", face_embedding=dummy_emb)
        world = WorldModel(entities=[entity], frame_id=1)

        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.08)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, source="test"))
        time.sleep(0.05)

        # 1. User says "yes"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "yes", "args": "", "raw_transcript": "yes"}, source="test"))
        time.sleep(0.05)

        # 2. User says "Ganesh"
        bus.publish(Event(topic=event_types.SPEECH_COMMAND, data={"action": "unmatched_text", "args": "Ganesh", "raw_transcript": "Ganesh."}, source="test"))
        time.sleep(0.1)

        # Check disk: data/faces/Ganesh/0.npy must exist!
        ganesh_file = os.path.join(db_dir, "Ganesh", "0.npy")
        assert os.path.exists(ganesh_file), f"Expected {ganesh_file} to exist on disk"
        
        # Load embedding from disk and verify it matches the exact original embedding
        saved_emb = np.load(ganesh_file)
        assert np.array_equal(saved_emb, dummy_emb), "Saved embedding on disk must match original track embedding"

        # Check that Person 1 was NOT created
        person1_dir = os.path.join(db_dir, "Person 1")
        assert not os.path.exists(person1_dir), f"Person 1 should NOT be created when name was supplied"

        controller.stop()
        bus.stop()

    def test_speak_pipeline_pauses_stt_throughout_playback_and_drain(self):
        """Verify SpeakPipeline pauses listening during playback and only resumes after sd.wait and drain."""
        from speech.manager.speak_pipeline import SpeakPipeline
        from speech.queue.priority_queue import SpeechPriorityQueue
        from speech.models.speech_priority import SpeechPriority
        from speech.models.audio_frame import AudioFrame

        pause_events = []
        resume_events = []

        class MockListenPipeline:
            def pause(self):
                pause_events.append(time.time())
            def resume(self):
                resume_events.append(time.time())

        class MockSynthesizer:
            def synthesize_stream(self, text, style):
                # Yield 1 audio frame of 1000 samples
                yield AudioFrame(
                    data=np.zeros(1000, dtype=np.int16),
                    sample_rate=16000,
                    channels=1,
                    dtype="int16",
                )

        class MockManager:
            def __init__(self):
                self._listen_pipeline = MockListenPipeline()

        manager = MockManager()
        priority_queue = SpeechPriorityQueue()
        pipeline = SpeakPipeline(
            manager=manager,
            synthesizer=MockSynthesizer(),
            priority_queue=priority_queue,
            output_device_id=None,
        )

        pipeline.start()
        priority_queue.put("Would you like me to save this person?", priority=SpeechPriority.NOTIFICATION)
        time.sleep(0.5)
        pipeline.stop()

        assert len(pause_events) >= 1
        assert len(resume_events) >= 1
        # Resume must have occurred after pause
        assert resume_events[0] >= pause_events[0]

    def test_speech_set_context_propagation_to_live_gate(self):
        """Verify SPEECH_SET_CONTEXT event through AppController correctly updates ConfidenceGate on live chain."""
        from core.app_controller import AppController
        from core.config.loader import load_config
        from speech.manager.speech_manager import SpeechManager
        from speech.stt.confidence_gate import ConfidenceGate
        from speech.manager.listen_pipeline import ListenPipeline

        gate = ConfidenceGate(expecting_name=False)
        assert gate.get_threshold("unmatched_text") == 0.50

        # Simulate live SpeechManager and ListenPipeline
        mock_listen = type("MockListenPipeline", (), {
            "_confidence_gate": gate,
            "set_expecting_name": lambda self, exp: gate.set_expecting_name(exp),
        })()
        mock_speech_manager = type("MockSpeechManager", (), {
            "_confidence_gate": gate,
            "_listen_pipeline": mock_listen,
            "set_expecting_name": lambda self, exp: [gate.set_expecting_name(exp), mock_listen.set_expecting_name(exp)],
        })()

        bus = EventBus()
        bus.start()

        config = load_config()
        app = AppController(config)
        app._event_bus = bus
        app._speech_manager = mock_speech_manager
        bus.subscribe(
            event_types.SPEECH_SET_CONTEXT,
            app._handle_speech_set_context,
            source="app_controller",
        )

        bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_name": True},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
        ))
        time.sleep(0.05)

        assert gate.expecting_name is True
        assert gate.get_threshold("unmatched_text") == 0.40

        bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_name": False},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
        ))
        time.sleep(0.05)

        assert gate.expecting_name is False
        assert gate.get_threshold("unmatched_text") == 0.50

        bus.stop()

    def test_multi_chunk_tts_complete_playback_lifecycle(self):
        """Verify all audio chunks are fully played before drain and STT resume."""
        from speech.manager.speak_pipeline import SpeakPipeline
        from speech.queue.priority_queue import SpeechPriorityQueue
        from speech.models.speech_priority import SpeechPriority
        from speech.models.audio_frame import AudioFrame

        events_timeline = []

        class MockListenPipeline:
            def pause(self):
                events_timeline.append(("pause", time.time()))
            def resume(self):
                events_timeline.append(("resume", time.time()))

        class MultiChunkSynthesizer:
            def synthesize_stream(self, text, style):
                for i in range(3):
                    events_timeline.append((f"chunk_{i+1}_yielded", time.time()))
                    yield AudioFrame(
                        data=np.zeros(500, dtype=np.int16),
                        sample_rate=16000,
                        channels=1,
                        dtype="int16",
                    )

        manager = type("MockManager", (), {"_listen_pipeline": MockListenPipeline()})()
        priority_queue = SpeechPriorityQueue()
        pipeline = SpeakPipeline(
            manager=manager,
            synthesizer=MultiChunkSynthesizer(),
            priority_queue=priority_queue,
            output_device_id=None,
        )

        pipeline.start()
        priority_queue.put("What should I call this person?", priority=SpeechPriority.NOTIFICATION)
        time.sleep(0.6)
        pipeline.stop()

        event_names = [e[0] for e in events_timeline]
        assert "pause" in event_names
        assert "chunk_1_yielded" in event_names
        assert "chunk_2_yielded" in event_names
        assert "chunk_3_yielded" in event_names
        assert "resume" in event_names

        # Resume must be the last event
        assert event_names[-1] == "resume"

    def test_exact_synthesis_and_complete_playback_for_registration_sentences(self):
        """Verify full synthesis duration, sample counts, and complete playback for all registration phrases."""
        from speech.tts.text_normalizer import TextNormalizer
        from speech.tts.pronunciation_dict import PronunciationDictionary
        from speech.tts.streaming_synthesizer import StreamingSynthesizer
        from speech.tts.pyttsx3_engine import Pyttsx3Engine
        from speech.tts.speech_cache import SpeechCache
        from speech.tts.emotional_profile import get_style_for_priority

        norm = TextNormalizer()
        pdict = PronunciationDictionary()
        cache = SpeechCache()
        engine = Pyttsx3Engine()
        engine.load()
        synth = StreamingSynthesizer(engine, cache, norm, pdict)
        style = get_style_for_priority(1)

        phrases = [
            ("Would you like me to save this person?", 30000, 1.5),
            ("What should I call this person?", 25000, 1.4),
            ("Ganesh saved.", 18000, 1.0),
        ]

        for phrase, min_samples, min_dur in phrases:
            chunks = list(synth.synthesize_stream(phrase, style))
            assert len(chunks) >= 1, f"Expected at least 1 chunk for {phrase}"
            combined = np.concatenate([c.data for c in chunks])
            sr = chunks[0].sample_rate
            dur = len(combined) / float(sr)

            assert len(combined) >= min_samples, f"{phrase}: Expected >= {min_samples} samples, got {len(combined)}"
            assert dur >= min_dur, f"{phrase}: Expected >= {min_dur}s duration, got {dur:.2f}s"

    def test_confirmation_confidence_threshold_resolution_and_checks(self):
        """Verify yes/yeah/ok at 0.33, 0.35, 0.42 are accepted under confirmation context and rejected outside."""
        from speech.stt.confidence_gate import ConfidenceGate
        from speech.stt.transcript import Transcript

        gate = ConfidenceGate(expecting_confirmation=True)
        assert gate.expecting_confirmation is True
        assert gate.get_threshold("yes") == 0.30
        assert gate.get_threshold("no") == 0.30
        assert gate.get_threshold("unmatched_text") == 0.30

        # Confidences from runtime log
        t_yes_33 = Transcript(text="Yes.", confidence=0.33, raw_text="Yes.")
        t_yeah_35 = Transcript(text="Yeah", confidence=0.35, raw_text="Yeah")
        t_yeah_42 = Transcript(text="Yeah.", confidence=0.42, raw_text="Yeah.")
        t_low_25 = Transcript(text="Yeah", confidence=0.25, raw_text="Yeah")

        assert gate.check(t_yes_33, action="yes").passed is True
        assert gate.check(t_yeah_35, action="yes").passed is True
        assert gate.check(t_yeah_42, action="yes").passed is True
        assert gate.check(t_low_25, action="yes").passed is False

        # Outside confirmation context: threshold returns to 0.50
        gate.expecting_confirmation = False
        assert gate.get_threshold("yes") == 0.50
        assert gate.check(t_yes_33, action="yes").passed is False
        assert gate.check(t_yeah_35, action="yes").passed is False
        assert gate.check(t_yeah_42, action="yes").passed is False

    def test_confirmation_context_propagation_via_event_bus(self):
        """Verify SPEECH_SET_CONTEXT accurately switches expecting_confirmation and expecting_name on ConfidenceGate."""
        from core.events.event_bus import EventBus
        from core.models.events import Event
        from core.events import event_types
        from core.models.enums import Priority
        from core.config.loader import load_config
        from core.app_controller import AppController
        from speech.stt.confidence_gate import ConfidenceGate

        bus = EventBus()
        bus.start()

        gate = ConfidenceGate()
        speech_mgr = type("MockSpeechMgr", (), {
            "_confidence_gate": gate,
            "_listen_pipeline": None,
            "set_expecting_confirmation": lambda s, v: gate.set_expecting_confirmation(v),
            "set_expecting_name": lambda s, v: gate.set_expecting_name(v),
        })()

        config = load_config()
        app = AppController(config)
        app._event_bus = bus
        app._speech_manager = speech_mgr
        bus.subscribe(
            event_types.SPEECH_SET_CONTEXT,
            app._handle_speech_set_context,
            source="app_controller",
        )

        # Enable confirmation context
        bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_confirmation": True, "expecting_name": False},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
        ))
        time.sleep(0.05)

        assert gate.expecting_confirmation is True
        assert gate.expecting_name is False
        assert gate.get_threshold("yes") == 0.30

        # Advance to name context
        bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_confirmation": False, "expecting_name": True},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
        ))
        time.sleep(0.05)

        assert gate.expecting_confirmation is False
        assert gate.expecting_name is True
        assert gate.get_threshold("yes") == 0.50
        assert gate.get_threshold("unmatched_text") == 0.40

        bus.stop()

    def test_end_to_end_confirmation_to_name_to_ganesh_save_flow(self, tmp_path):
        """Verify full flow: unknown face -> PROMPTING_SAVE -> yeah (0.35) -> PROMPTING_NAME -> Ganesh (0.46) -> save."""
        from core.events.event_bus import EventBus
        from core.models.events import Event
        from core.events import event_types
        from core.models.enums import Priority
        from core.config.loader import load_config
        from core.app_controller import AppController
        from core.logic.logic_controller import LogicController
        from core.logic.face_registration_tracker import FaceRegistrationTracker, TrackerState
        from vision.pipeline.perception_fusion import WorldModel, Entity
        from speech.stt.confidence_gate import ConfidenceGate
        from speech.stt.transcript import Transcript

        bus = EventBus()
        bus.start()

        faces_dir = str(tmp_path / "faces")
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.1,
            grace_period_s=5.0,
            name_timeout_s=15.0,
            face_db_dir=faces_dir,
        )
        logic = LogicController(event_bus=bus)
        logic._face_tracker = tracker
        logic.start()

        gate = ConfidenceGate()
        config = load_config()
        app = AppController(config)
        app._event_bus = bus
        speech_mgr = type("MockSpeechMgr", (), {
            "_confidence_gate": gate,
            "set_expecting_confirmation": lambda s, v: gate.set_expecting_confirmation(v),
            "set_expecting_name": lambda s, v: gate.set_expecting_name(v),
            "speak": lambda s, text, priority=1: None,
        })()
        app._speech_manager = speech_mgr
        bus.subscribe(
            event_types.SPEECH_SET_CONTEXT,
            app._handle_speech_set_context,
            source="app_controller",
        )
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            app._handle_logic_announce,
            source="app_controller",
        )

        published_events = []
        bus.subscribe(event_types.LOGIC_ANNOUNCE, lambda e: published_events.append(e))
        bus.subscribe(event_types.VISION_SAVE_FACE, lambda e: published_events.append(e))
        bus.subscribe(event_types.SPEECH_SET_CONTEXT, lambda e: published_events.append(e))

        # 1. Unknown face detected with original embedding
        original_embedding = np.random.randn(512).astype(np.float32)
        world = WorldModel(
            timestamp=time.time(),
            entities=[
                Entity(track_id=1, cls="person", bbox=(0, 0, 100, 100), confidence=0.9, face_name="Unknown", face_embedding=original_embedding)
            ]
        )
        tracker.update_world(world)
        time.sleep(0.15)
        # Second frame triggers prompt
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, priority=Priority.INFORMATIONAL, source="vision"))
        time.sleep(0.05)

        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE
        assert gate.expecting_confirmation is True
        assert gate.expecting_name is False

        # 2. User confirms with "Yeah" at confidence 0.35
        gate_res = gate.check(Transcript(text="Yeah", confidence=0.35, raw_text="Yeah"), action="yes")
        assert gate_res.passed is True

        bus.publish(Event(
            topic=event_types.SPEECH_COMMAND,
            data={"action": "yes", "args": "", "raw_transcript": "Yeah"},
            priority=Priority.INFORMATIONAL,
            source="app_controller",
        ))
        time.sleep(0.05)

        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_NAME
        assert gate.expecting_confirmation is False
        assert gate.expecting_name is True

        # 3. User says name "Ganesh" at confidence 0.46
        gate_res_name = gate.check(Transcript(text="Ganesh.", confidence=0.46, raw_text="Ganesh."), action="unmatched_text")
        assert gate_res_name.passed is True

        bus.publish(Event(
            topic=event_types.SPEECH_COMMAND,
            data={"action": "unmatched_text", "args": "Ganesh", "raw_transcript": "Ganesh."},
            priority=Priority.INFORMATIONAL,
            source="app_controller",
        ))
        time.sleep(0.05)

        # 4. Verify save event and speech context reset
        save_events = [e for e in published_events if e.topic == event_types.VISION_SAVE_FACE]
        assert len(save_events) == 1
        assert save_events[0].data["name"] == "Ganesh"
        assert np.array_equal(save_events[0].data["embedding"], original_embedding)

        assert gate.expecting_confirmation is False
        assert gate.expecting_name is False

        logic.stop()
        bus.stop()

    def test_save_prompt_declined_by_no(self, tmp_path):
        """Verify user saying 'no' at PROMPTING_SAVE marks encounter declined and resets context."""
        from core.events.event_bus import EventBus
        from core.models.events import Event
        from core.events import event_types
        from core.models.enums import Priority
        from core.config.loader import load_config
        from core.app_controller import AppController
        from core.logic.logic_controller import LogicController
        from core.logic.face_registration_tracker import FaceRegistrationTracker, TrackerState
        from vision.pipeline.perception_fusion import WorldModel, Entity
        from speech.stt.confidence_gate import ConfidenceGate

        bus = EventBus()
        bus.start()

        faces_dir = str(tmp_path / "faces")
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.1,
            grace_period_s=5.0,
            name_timeout_s=15.0,
            face_db_dir=faces_dir,
        )
        logic = LogicController(event_bus=bus)
        logic._face_tracker = tracker
        logic.start()

        gate = ConfidenceGate()
        config = load_config()
        app = AppController(config)
        app._event_bus = bus
        speech_mgr = type("MockSpeechMgr", (), {
            "_confidence_gate": gate,
            "set_expecting_confirmation": lambda s, v: gate.set_expecting_confirmation(v),
            "set_expecting_name": lambda s, v: gate.set_expecting_name(v),
            "speak": lambda s, text, priority=1: None,
        })()
        app._speech_manager = speech_mgr
        bus.subscribe(
            event_types.SPEECH_SET_CONTEXT,
            app._handle_speech_set_context,
            source="app_controller",
        )
        bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            app._handle_logic_announce,
            source="app_controller",
        )

        world = WorldModel(
            timestamp=time.time(),
            entities=[Entity(track_id=1, cls="person", bbox=(0, 0, 100, 100), confidence=0.9, face_name="Unknown", face_embedding=np.zeros(512))]
        )
        tracker.update_world(world)
        time.sleep(0.15)
        bus.publish(Event(topic=event_types.VISION_WORLD_UPDATE, data={"world": world}, priority=Priority.INFORMATIONAL, source="vision"))
        time.sleep(0.05)

        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE

        # User says "no"
        bus.publish(Event(
            topic=event_types.SPEECH_COMMAND,
            data={"action": "no", "args": "", "raw_transcript": "No"},
            priority=Priority.INFORMATIONAL,
            source="app_controller",
        ))
        time.sleep(0.05)

        assert tracker.get_active_prompt_state() is None
        assert gate.expecting_confirmation is False
        assert gate.expecting_name is False

        logic.stop()
        bus.stop()

    def test_active_registration_encounter_survives_temporary_face_loss(self, tmp_path):
        """Verify active registration dialog and original embedding survive temporary face absence and track ID changes."""
        from core.logic.face_registration_tracker import FaceRegistrationTracker, TrackerState
        from vision.pipeline.perception_fusion import WorldModel, Entity

        faces_dir = str(tmp_path / "faces")
        tracker = FaceRegistrationTracker(
            visibility_threshold_s=0.1,
            grace_period_s=1.0,
            name_timeout_s=15.0,
            face_db_dir=faces_dir,
        )

        original_embedding = np.ones(512, dtype=np.float32)
        world_init = WorldModel(
            timestamp=time.time(),
            entities=[Entity(track_id=1, cls="person", bbox=(0, 0, 100, 100), confidence=0.9, face_name="Unknown", face_embedding=original_embedding)]
        )
        tracker.update_world(world_init)
        time.sleep(0.15)
        tid = tracker.update_world(world_init)
        assert tid == 1
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE

        # Face disappears for 2.5 seconds (e.g. during TTS playback)
        time.sleep(0.1)
        empty_world = WorldModel(timestamp=time.time(), entities=[])
        tracker.update_world(empty_world)

        # Face reappears with track_id=2 (new camera track) without embedding
        world_reacquired = WorldModel(
            timestamp=time.time(),
            entities=[Entity(track_id=2, cls="person", bbox=(10, 10, 100, 100), confidence=0.9, face_name="Unknown", face_embedding=None)]
        )
        tracker.update_world(world_reacquired)

        # Dialog and original embedding must remain active
        assert tracker.get_active_prompt_state() == TrackerState.PROMPTING_SAVE
        assert tracker.get_active_track_id() == 2
        assert np.array_equal(tracker.get_active_embedding(), original_embedding)

