"""Logic Controller — event-driven decision engine.

The central "brain" of SIMON. Subscribes to vision and navigation events,
applies safety checks, ranks detections by priority, generates
announcements, and publishes speech and action events.

Event flow::

    vision.world_update → LogicController → logic.announce (to TTS)
    safety.hazard       → LogicController → safety.emergency (immediate TTS)
    speech.command      → LogicController → actions (navigate, read text, etc.)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.logic.priority_engine import PriorityEngine
from core.logic.spatial_reasoner import SpatialReasoner
from core.logic.language_generator import LanguageGenerator
from core.memory.announcement_tracker import AnnouncementTracker
from core.logic.face_registration_tracker import FaceRegistrationTracker, TrackerState
from core.logic.known_person_tracker import KnownPersonTracker, KNOWN_PERSON_REANNOUNCE_TIMEOUT_S
from core.visual_context.visual_information_manager import VisualInformationManager, VisualEvent
from core.visual_context.classifier import VisualPriority
from vision.pipeline.perception_fusion import WorldModel, Entity

logger = logging.getLogger("simon.core.logic")


class LogicController:
    """Event-driven decision engine.

    Subscribes to vision and safety events, processes world updates,
    and generates speech announcements.

    Parameters
    ----------
    event_bus : EventBus
        For subscribing to events and publishing announcements.
    priority_engine : PriorityEngine, optional
        For ranking entity importance.
    spatial_reasoner : SpatialReasoner, optional
        For generating spatial descriptions.
    language_generator : LanguageGenerator, optional
        For generating speech text.
    announcement_tracker : AnnouncementTracker, optional
        For deduplicating announcements.
    known_person_tracker : KnownPersonTracker, optional
        For managing arrival announcements of recognized individuals.
    visual_info_manager : VisualInformationManager, optional
        For context-aware visual information classification, immediate TTS, and post-TTS Ollama intelligence.
    max_announcements_per_frame : int
        Maximum announcements per world update.
    """

    def __init__(
        self,
        event_bus: EventBus,
        priority_engine: Optional[PriorityEngine] = None,
        spatial_reasoner: Optional[SpatialReasoner] = None,
        language_generator: Optional[LanguageGenerator] = None,
        announcement_tracker: Optional[AnnouncementTracker] = None,
        known_person_tracker: Optional[KnownPersonTracker] = None,
        visual_info_manager: Optional[VisualInformationManager] = None,
        max_announcements_per_frame: int = 1,
        vision_pipeline=None,
    ) -> None:
        self._event_bus = event_bus
        self._vision_pipeline = vision_pipeline
        self._priority = priority_engine or PriorityEngine()
        self._spatial = spatial_reasoner or SpatialReasoner()
        self._language = language_generator or LanguageGenerator()
        self._tracker = announcement_tracker or AnnouncementTracker()
        self._face_tracker = FaceRegistrationTracker()
        self._known_person_tracker = known_person_tracker or KnownPersonTracker()
        self._visual_info_manager = visual_info_manager or VisualInformationManager()
        self._max_announcements = max_announcements_per_frame
        self._latest_world: Optional[WorldModel] = None
        self._running = False

    @property
    def visual_info_manager(self) -> VisualInformationManager:
        return self._visual_info_manager

    def start(self) -> None:
        """Subscribe to events and start processing."""
        self._running = True
        self._event_bus.subscribe(
            event_types.VISION_WORLD_UPDATE,
            self._on_world_update,
            source="logic_controller",
        )
        self._event_bus.subscribe(
            event_types.VISION_HAZARD_DETECTED,
            self._on_hazard,
            priority_filter=Priority.SAFETY_CRITICAL,
            source="logic_controller",
        )
        self._event_bus.subscribe(
            event_types.VISION_TEXT_WARNING,
            self._on_text_warning,
            priority_filter=Priority.SAFETY_CRITICAL,
            source="logic_controller",
        )
        self._event_bus.subscribe(
            event_types.SPEECH_COMMAND,
            self._on_voice_command,
            source="logic_controller",
        )
        self._event_bus.subscribe(
            event_types.SPEECH_PLAYBACK_COMPLETE,
            self._on_speech_playback_complete,
            source="logic_controller",
        )
        logger.info("Logic controller started")

    def stop(self) -> None:
        """Stop the logic controller."""
        self._running = False
        logger.info("Logic controller stopped")

    @property
    def latest_world(self) -> Optional[WorldModel]:
        return self._latest_world

    def process_world(self, world: WorldModel) -> list[str]:
        """Process a world update and return announcements.

        This is the core decision method. It:
        1. Ranks entities by priority.
        2. Filters already-announced entities.
        3. Generates natural language announcements.
        4. Returns text ready for TTS.
        """
        self._latest_world = world

        # Rank entities by priority
        ranked = self._priority.rank_entities(world)

        # Generate announcements (limited per frame)
        announcements: list[str] = []
        
        for entity in ranked:
            if len(announcements) >= self._max_announcements:
                break

            # Check if already announced recently
            key = self._announcement_key(entity)
            if self._tracker.was_announced(key):
                continue

            text = self._language.announce_entity(entity)
            if text:
                announcements.append(text)
                self._tracker.mark_announced(key)

        return announcements

    def get_scene_description(self) -> str:
        """Generate a scene description for 'what do you see?' queries."""
        if self._latest_world is None:
            return "I'm not seeing anything right now"
        return self._language.scene_summary(self._latest_world)

    # ── Event Handlers ───────────────────────────────────────────────

    def _on_world_update(self, event: Event) -> None:
        """Handle vision.world_update events."""
        if not self._running:
            return

        # Retrieve WorldModel and trigger trackers
        world = event.data.get("world") or (self._vision_pipeline.latest_world if self._vision_pipeline else None) or self._latest_world
        if world:
            # 1. Check known person arrivals
            for entity in world.entities:
                if entity.face_name and entity.face_name != "Unknown":
                    should_announce = self._known_person_tracker.process_detection(entity.face_name)
                    if should_announce:
                        announcement_text = self._language.known_person_arrival(
                            name=entity.face_name,
                            position=entity.position,
                            distance=entity.distance,
                        )
                        logger.info("[KNOWN_PERSON] Publishing arrival announcement: %r", announcement_text)
                        self._event_bus.publish(Event(
                            topic=event_types.LOGIC_ANNOUNCE,
                            data={"text": announcement_text, "priority": Priority.INFORMATIONAL},
                            priority=Priority.INFORMATIONAL,
                            source="logic_controller",
                            correlation_id=event.correlation_id,
                        ))

            # 2. Context-Aware Visual Information System
            visual_events = self._visual_info_manager.process_world(world)
            for ve in visual_events:
                if ve.spoken_response:
                    pri = Priority.SAFETY_CRITICAL if ve.priority == VisualPriority.P0_SAFETY_CRITICAL else Priority.INFORMATIONAL
                    self._event_bus.publish(Event(
                        topic=event_types.LOGIC_ANNOUNCE,
                        data={
                            "text": ve.spoken_response,
                            "priority": pri,
                            "event_key": ve.key,
                        },
                        priority=pri,
                        source="logic_controller",
                        correlation_id=event.correlation_id,
                    ))

            # 3. Check unknown face registration prompts
            prompt_track_id = self._face_tracker.update_world(world)
            if prompt_track_id is not None:
                logger.info("[FACE_REG] Sending save-question to SpeechManager")
                logger.info("[FACE_REG] Awaiting user confirmation")
                logger.info("[CONFIRM_TRACE] LogicController publishing expecting_confirmation=True for track_id=%s", prompt_track_id)
                self._event_bus.publish(Event(
                    topic=event_types.SPEECH_SET_CONTEXT,
                    data={"expecting_confirmation": True, "expecting_name": False},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))
                self._event_bus.publish(Event(
                    topic=event_types.LOGIC_ANNOUNCE,
                    data={"text": "Would you like me to save this person?", "priority": Priority.INFORMATIONAL},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))

            # 4. Check for name-prompt timeout → save anonymously
            if self._face_tracker.check_name_timeout():
                self._save_anonymous(event.correlation_id)

    def _on_speech_playback_complete(self, event: Event) -> None:
        """Handle SPEECH_PLAYBACK_COMPLETE to notify VisualInformationManager post-TTS."""
        event_key = event.data.get("event_key")
        spoken_text = event.data.get("text")
        success = event.data.get("success", True)
        self._visual_info_manager.on_tts_completed(event_key=event_key, spoken_text=spoken_text, success=success)

    def _on_hazard(self, event: Event) -> None:
        """Handle safety-critical hazard events — immediate TTS."""
        if not self._running:
            return

        cls = event.data.get("cls", "object")
        position = event.data.get("position", "")
        distance = event.data.get("distance", "")

        entity = Entity(
            track_id=event.data.get("track_id", 0),
            cls=cls,
            bbox=(0, 0, 0, 0),
            confidence=1.0,
            position=position,
            distance=distance,
            hazard_level=event.data.get("hazard_level"),
        )

        text = self._language.announce_entity(entity)
        if text:
            self._event_bus.publish(Event(
                topic=event_types.SAFETY_EMERGENCY,
                data={"message": text, "priority": Priority.EMERGENCY},
                priority=Priority.EMERGENCY,
                source="logic_controller",
                correlation_id=event.correlation_id,
            ))
            logger.warning("Hazard announcement: %s", text)

    def _on_text_warning(self, event: Event) -> None:
        """Handle OCR text warnings."""
        if not self._running:
            return

        ocr_text = event.data.get("text", "")
        if not ocr_text:
            return

        key = f"ocr_warning:{ocr_text.lower()}"
        if self._tracker.was_announced(key):
            return

        # Announce the warning
        spoken_text = f"Warning: {ocr_text}"
        self._event_bus.publish(Event(
            topic=event_types.SPEECH_COMMAND,
            data={"text": spoken_text},
            priority=Priority.SAFETY_CRITICAL,
            source="logic_controller",
            correlation_id=event.correlation_id,
        ))
        self._tracker.mark_announced(key)

    def _on_voice_command(self, event: Event) -> None:
        """Handle speech.command events — user voice commands."""
        if not self._running:
            return

        action = event.data.get("action", "")
        args = event.data.get("args", "")
        raw_transcript = event.data.get("raw_transcript", "")

        logger.info(
            "[EVENT] SPEECH_COMMAND received by LogicController: action=%r, args=%r, raw=%r",
            action, args, raw_transcript,
        )

        # Check for active face registration prompt FIRST
        prompt_state = self._face_tracker.get_active_prompt_state()
        active_track_id = self._face_tracker.get_active_track_id()

        logger.info(
            "[NAME_TRACE] LogicController received command: action=%r, args=%r, raw=%r, prompt_state=%s, track_id=%s",
            action, args, raw_transcript, prompt_state, active_track_id,
        )
        logger.info(
            "[CONFIRM_TRACE] LogicController received command: action=%r, args=%r, raw=%r, prompt_state=%s, track_id=%s",
            action, args, raw_transcript, prompt_state, active_track_id,
        )

        # ── State: PROMPTING_SAVE (waiting for yes/no) ───────────────
        if prompt_state == TrackerState.PROMPTING_SAVE:
            # Check for YES
            if action == "yes" or (action == "unmatched_text" and args.lower() in ("yes", "yeah", "yep", "sure", "okay", "ok")):
                logger.info("[FACE_REG] User accepted registration for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[NAME_TRACE] Transitioning PROMPTING_SAVE -> PROMPTING_NAME for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[CONFIRM_TRACE] User confirmed save for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[CONFIRM_TRACE] State transition: PROMPTING_SAVE -> PROMPTING_NAME for track_id=%s", self._face_tracker.get_active_track_id())
                self._face_tracker.advance_to_name_prompt()
                self._event_bus.publish(Event(
                    topic=event_types.SPEECH_SET_CONTEXT,
                    data={"expecting_confirmation": False, "expecting_name": True},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))
                self._event_bus.publish(Event(
                    topic=event_types.LOGIC_ANNOUNCE,
                    data={"text": "What should I call this person?", "priority": Priority.INFORMATIONAL},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))
                return
            # Check for NO
            elif action == "no" or (action == "unmatched_text" and args.lower() in ("no", "nope", "no thanks", "cancel", "don't save", "do not save", "dont save")):
                logger.info(
                    "[FACE_REG] User declined save for track_id=%s",
                    self._face_tracker.get_active_track_id(),
                )
                logger.info("[NAME_TRACE] Save declined for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[CONFIRM_TRACE] User declined save for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[CONFIRM_TRACE] State transition: PROMPTING_SAVE -> DECLINED for track_id=%s", self._face_tracker.get_active_track_id())
                self._face_tracker.mark_declined()
                self._event_bus.publish(Event(
                    topic=event_types.SPEECH_SET_CONTEXT,
                    data={"expecting_confirmation": False, "expecting_name": False},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))
                return
            # Other commands during prompting: ignore for registration, pass through

        # ── State: PROMPTING_NAME (waiting for a name) ───────────────
        elif prompt_state == TrackerState.PROMPTING_NAME:
            # "no_name" → save anonymously
            if action == "no_name" or (action == "unmatched_text" and args.lower() in ("no name", "none", "without a name", "don't name", "skip")):
                logger.info("[FACE_REG] User requested anonymous save for track_id=%s", self._face_tracker.get_active_track_id())
                logger.info("[NAME_TRACE] Explicit anonymous save requested for track_id=%s", self._face_tracker.get_active_track_id())
                self._save_anonymous(event.correlation_id)
                return

            # "no" / "cancel" → user changed their mind and declines saving
            if action == "no" or (action == "unmatched_text" and args.lower() in ("no", "nope", "cancel", "stop", "don't save", "do not save", "dont save")):
                logger.info(
                    "[FACE_REG] User declined name for track_id=%s",
                    self._face_tracker.get_active_track_id(),
                )
                logger.info("[NAME_TRACE] Name declined for track_id=%s", self._face_tracker.get_active_track_id())
                self._face_tracker.mark_declined()
                self._event_bus.publish(Event(
                    topic=event_types.SPEECH_SET_CONTEXT,
                    data={"expecting_confirmation": False, "expecting_name": False},
                    priority=Priority.INFORMATIONAL,
                    source="logic_controller",
                    correlation_id=event.correlation_id,
                ))
                return

            # Spoken name
            if action == "unmatched_text":
                name = args.strip().title() if args else ""
            else:
                # Might be a name that matched another command or phrase; use raw transcript or action
                name = (raw_transcript or args or action).strip().title()

            if name:
                logger.info("[FACE_REG] Name provided: %r for track_id=%s", name, self._face_tracker.get_active_track_id())
                logger.info("[NAME_TRACE] Spoken name accepted: %r for track_id=%s", name, self._face_tracker.get_active_track_id())
                self._save_named(name, event.correlation_id)
                return

        # ── Normal command routing ───────────────────────────────────
        query_candidates = (raw_transcript or args or action or "").lower()
        if any(qw in query_candidates for qw in ("what is around", "what do you see", "what did you see", "what was that sign", "read the sign", "what does that board say", "restaurant", "bus stop", "where does this road go", "what area", "what place")):
            answer = self._visual_info_manager.answer_user_query(raw_transcript or args or action)
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_ANNOUNCE,
                data={"text": answer, "priority": Priority.INFORMATIONAL},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=event.correlation_id,
            ))
            return

        if action in ("describe", "what_do_you_see", "look"):
            description = self.get_scene_description()
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_ANNOUNCE,
                data={"text": description, "priority": Priority.INFORMATIONAL},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=event.correlation_id,
            ))

    # ── Face Registration Helpers ────────────────────────────────────

    def _save_named(self, name: str, correlation_id: Optional[str] = None) -> None:
        """Save the currently tracked face with a user-provided name."""
        embedding = self._face_tracker.get_active_embedding()
        track_id = self._face_tracker.get_active_track_id()
        logger.info("[NAME_TRACE] save_named: %r for track_id=%s", name, track_id)
        logger.info("[NAME_TRACE] embedding source track_id: %s", track_id)
        logger.info("[NAME_TRACE] embedding shape: %s", getattr(embedding, "shape", None) if embedding is not None else None)
        logger.info(
            "[NAME_TRACE] Save method selected: _save_named(%r) for track_id=%s, embedding_present=%s",
            name, track_id, (embedding is not None),
        )
        if embedding is not None:
            logger.info("[FACE_REG] Save requested: track_id=%s, name=%r", track_id, name)
            logger.info("[FACE_REG] Named save: %r for track_id=%s", name, track_id)
            self._event_bus.publish(Event(
                topic=event_types.VISION_SAVE_FACE,
                data={"track_id": track_id, "embedding": embedding, "name": name},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=correlation_id,
            ))
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_ANNOUNCE,
                data={"text": f"{name} saved.", "priority": Priority.INFORMATIONAL},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=correlation_id,
            ))
        self._face_tracker.mark_completed()
        self._event_bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_confirmation": False, "expecting_name": False},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
            correlation_id=correlation_id,
        ))

    def _save_anonymous(self, correlation_id: Optional[str] = None) -> None:
        """Save the currently tracked face with an auto-generated name."""
        anon_name = self._face_tracker.get_next_anonymous_name()
        embedding = self._face_tracker.get_active_embedding()
        track_id = self._face_tracker.get_active_track_id()
        logger.info(
            "[NAME_TRACE] Save method selected: _save_anonymous(%r) for track_id=%s, embedding_present=%s",
            anon_name, track_id, (embedding is not None),
        )
        if embedding is not None:
            logger.info("[FACE_REG] Save requested: track_id=%s, name=%r", track_id, anon_name)
            logger.info("[FACE_REG] Anonymous save: %r for track_id=%s", anon_name, track_id)
            self._event_bus.publish(Event(
                topic=event_types.VISION_SAVE_FACE,
                data={"track_id": track_id, "embedding": embedding, "name": anon_name},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=correlation_id,
            ))
            self._event_bus.publish(Event(
                topic=event_types.LOGIC_ANNOUNCE,
                data={"text": f"{anon_name} saved.", "priority": Priority.INFORMATIONAL},
                priority=Priority.INFORMATIONAL,
                source="logic_controller",
                correlation_id=correlation_id,
            ))
        self._face_tracker.mark_completed()
        self._event_bus.publish(Event(
            topic=event_types.SPEECH_SET_CONTEXT,
            data={"expecting_confirmation": False, "expecting_name": False},
            priority=Priority.INFORMATIONAL,
            source="logic_controller",
            correlation_id=correlation_id,
        ))

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _announcement_key(entity: Entity) -> str:
        """Generate a deduplication key for an entity."""
        if entity.face_name:
            return f"face:{entity.face_name}"
        return f"{entity.cls}:{entity.position or 'unknown'}"
