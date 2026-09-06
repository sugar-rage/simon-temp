"""Visual Information Manager - orchestrates the Context-Aware Visual Information System.

ENFORCES THE CRITICAL ORDERING:
PERCEPTION -> DECISION -> TTS START -> TTS COMPLETE -> OLLAMA

Ollama NEVER delays the initial spoken response.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

from vision.pipeline.perception_fusion import WorldModel, Entity
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

logger = logging.getLogger("simon.core.visual_context.manager")


@dataclass
class VisualEvent:
    """An actionable visual event requiring TTS and post-TTS Ollama intelligence."""
    key: str
    classification: ClassificationResult
    spoken_response: Optional[str]
    priority: VisualPriority
    raw_ocr: List[str]
    detected_objects: List[str]
    faces: List[str]
    direction: Optional[str]
    distance: Optional[str]
    timestamp: float = field(default_factory=time.time)


class VisualInformationManager:
    """Coordinates classification, immediate TTS, post-TTS Ollama analysis, and UI state."""

    def __init__(
        self,
        classifier: Optional[VisualInfoClassifier] = None,
        event_tracker: Optional[VisualEventTracker] = None,
        context_memory: Optional[ContextualVisualMemory] = None,
        ollama_client: Optional[OllamaClient] = None,
        ollama_enabled: bool = True,
    ):
        self._classifier = classifier or VisualInfoClassifier()
        self._tracker = event_tracker or VisualEventTracker()
        self._memory = context_memory or ContextualVisualMemory()
        self._ollama = ollama_client or OllamaClient(enabled=ollama_enabled)

        # UI Overlay state cache
        self._ui_state: Dict[str, Any] = {
            "objects": [],
            "faces": [],
            "ocr": [],
            "classification": "NORMAL",
            "direction": "Ahead",
            "distance": "",
            "tts_text": "",
            "tts_status": "Idle",
            "ollama_status": "Idle",
            "last_updated": time.time(),
        }

        # Pending events awaiting TTS completion: {key: VisualEvent}
        self._pending_tts_events: Dict[str, VisualEvent] = {}

    @property
    def memory(self) -> ContextualVisualMemory:
        return self._memory

    @property
    def ollama_client(self) -> OllamaClient:
        return self._ollama

    def process_world(self, world: WorldModel) -> List[VisualEvent]:
        """Evaluate perception WorldModel and extract actionable visual events."""
        now = time.time()
        events_to_announce: List[VisualEvent] = []

        detected_objects = [e.cls for e in world.entities if e.cls and e.cls != "person"]
        faces = [e.face_name for e in world.entities if e.face_name]
        ocr_strings = [ocr.text.strip() for ocr in world.ocr_texts if ocr.text and ocr.text.strip()]

        # Update UI base detections
        self._ui_state["objects"] = detected_objects
        self._ui_state["faces"] = faces
        self._ui_state["ocr"] = ocr_strings
        self._ui_state["last_updated"] = now

        # 1. Process OCR texts
        for ocr_item in world.ocr_texts:
            text = ocr_item.text.strip()
            if not text:
                continue

            pos = getattr(ocr_item, "position", None) or "center"
            dist = getattr(ocr_item, "distance", None) or "medium"

            res = self._classifier.classify_ocr(text=text, position=pos, distance=dist)
            self._tracker.register_seen(res.canonical_key, text, res.priority.value, now=now)

            if not self._tracker.should_process(res.canonical_key, now=now):
                continue

            # Update UI classification
            self._ui_state["classification"] = res.priority.value
            self._ui_state["direction"] = pos.capitalize() if pos else "Ahead"
            self._ui_state["distance"] = dist.capitalize() if dist else ""

            # Check if this event warrants immediate TTS (P0 / P1)
            if res.immediate_tts_response:
                logger.info("[VISUAL_INFO] detected OCR=%r", text)
                logger.info("[VISUAL_INFO] classification=%s", res.priority.value)
                logger.info("[VISUAL_INFO] response=%r", res.immediate_tts_response)

                event = VisualEvent(
                    key=res.canonical_key,
                    classification=res,
                    spoken_response=res.immediate_tts_response,
                    priority=res.priority,
                    raw_ocr=[text],
                    detected_objects=detected_objects,
                    faces=faces,
                    direction=pos,
                    distance=dist,
                    timestamp=now,
                )
                self._tracker.mark_tts_started(res.canonical_key, now=now)
                self._pending_tts_events[res.canonical_key] = event

                self._ui_state["tts_text"] = res.immediate_tts_response
                self._ui_state["tts_status"] = "Speaking"
                self._ui_state["ollama_status"] = "Waiting"

                events_to_announce.append(event)
            else:
                # Normal OCR or Ambiguous non-safety text -> No immediate speech
                if res.is_ambiguous:
                    logger.info("[VISUAL_INFO] OCR ambiguous=%r", text)
                
                # Directly dispatch post-frame background Ollama analysis to store context
                self._tracker.mark_ollama_dispatched(res.canonical_key)
                self._ui_state["tts_status"] = "Completed"
                self._ui_state["ollama_status"] = "Processing"
                self._dispatch_ollama_analysis(
                    key=res.canonical_key,
                    ocr_text=text,
                    objects=detected_objects,
                    faces=faces,
                    direction=pos,
                    distance=dist,
                    classification=res.priority.value,
                    spoken_response=None,
                    is_ambiguous=res.is_ambiguous,
                )

        # 2. Process hazardous entities (e.g. dog / animals)
        for entity in world.entities:
            ent_res = self._classifier.classify_entity(
                entity_cls=entity.cls,
                position=entity.position,
                distance=entity.distance,
                confidence=entity.confidence,
            )
            if ent_res and ent_res.immediate_tts_response:
                self._tracker.register_seen(ent_res.canonical_key, ent_res.raw_text, ent_res.priority.value, now=now)
                if self._tracker.should_process(ent_res.canonical_key, now=now):
                    logger.info("[VISUAL_INFO] detected entity=%r", entity.cls)
                    logger.info("[VISUAL_INFO] classification=%s", ent_res.priority.value)
                    logger.info("[VISUAL_INFO] response=%r", ent_res.immediate_tts_response)

                    event = VisualEvent(
                        key=ent_res.canonical_key,
                        classification=ent_res,
                        spoken_response=ent_res.immediate_tts_response,
                        priority=ent_res.priority,
                        raw_ocr=[],
                        detected_objects=[entity.cls],
                        faces=faces,
                        direction=entity.position,
                        distance=entity.distance,
                        timestamp=now,
                    )
                    self._tracker.mark_tts_started(ent_res.canonical_key, now=now)
                    self._pending_tts_events[ent_res.canonical_key] = event

                    self._ui_state["classification"] = ent_res.priority.value
                    self._ui_state["tts_text"] = ent_res.immediate_tts_response
                    self._ui_state["tts_status"] = "Speaking"
                    self._ui_state["ollama_status"] = "Waiting"

                    events_to_announce.append(event)

        return events_to_announce

    def on_tts_completed(self, event_key: Optional[str] = None, spoken_text: Optional[str] = None, success: bool = True) -> None:
        """Called STRICTLY AFTER TTS playback and acoustic drain have completed.

        This is the critical gate where Ollama analysis is launched.
        """
        now = time.time()
        logger.info("[TTS] completed response=%r", spoken_text)
        self._ui_state["tts_status"] = "Completed"

        # Find matching event
        event: Optional[VisualEvent] = None
        if event_key and event_key in self._pending_tts_events:
            event = self._pending_tts_events.pop(event_key)
        elif spoken_text:
            for k, ev in list(self._pending_tts_events.items()):
                if ev.spoken_response == spoken_text:
                    event = self._pending_tts_events.pop(k)
                    event_key = k
                    break

        if event_key:
            self._tracker.mark_tts_completed(event_key, now=now)
            self._tracker.mark_ollama_dispatched(event_key)

        # Launch Ollama Post-TTS processing
        self._ui_state["ollama_status"] = "Processing"
        self._dispatch_ollama_analysis(
            key=event_key or "TTS_EVENT",
            ocr_text=", ".join(event.raw_ocr) if event and event.raw_ocr else (spoken_text or ""),
            objects=event.detected_objects if event else self._ui_state["objects"],
            faces=event.faces if event else self._ui_state["faces"],
            direction=event.direction if event else self._ui_state["direction"],
            distance=event.distance if event else self._ui_state["distance"],
            classification=event.priority.value if event else "SAFETY_CRITICAL",
            spoken_response=spoken_text or (event.spoken_response if event else ""),
            is_ambiguous=event.classification.is_ambiguous if event else False,
        )

    def _dispatch_ollama_analysis(
        self,
        key: str,
        ocr_text: str,
        objects: List[str],
        faces: List[str],
        direction: Optional[str],
        distance: Optional[str],
        classification: str,
        spoken_response: Optional[str],
        is_ambiguous: bool = False,
    ) -> None:
        """Send visual payload to Ollama for background context extraction."""
        payload = {
            "ocr_text": ocr_text,
            "detected_objects": objects,
            "faces": faces,
            "direction": direction or "ahead",
            "distance": distance or "medium",
            "classification": classification,
            "spoken_response": spoken_response,
            "is_ambiguous": is_ambiguous,
            "timestamp": time.time(),
        }

        def _on_ollama_done(extracted: Dict[str, Any]) -> None:
            self._tracker.mark_ollama_completed(key)
            memory_entry = VisualMemoryEntry(
                timestamp=time.time(),
                raw_ocr=[ocr_text] if ocr_text else [],
                detected_objects=objects,
                faces=faces,
                classification=classification,
                spoken_response=spoken_response,
                direction=direction,
                distance=distance,
                event_type=extracted.get("event"),
                place_name=extracted.get("place_name"),
                category=extracted.get("category"),
                recommended_action=extracted.get("recommended_action"),
                summary=extracted.get("summary", ""),
                raw_json=extracted,
            )
            self._memory.add_entry(memory_entry)
            logger.info("[OLLAMA] context updated: %s", extracted.get("summary") or extracted.get("event"))
            src = extracted.get("_source", "ollama")
            if src == "fallback" and not getattr(self._ollama, "_enabled", True):
                self._ui_state["ollama_status"] = "Unavailable"
            elif src == "fallback":
                self._ui_state["ollama_status"] = "Completed (Fallback)"
            else:
                self._ui_state["ollama_status"] = "Completed"

        self._ollama.analyze_post_tts_async(payload, on_complete=_on_ollama_done)

    def answer_user_query(self, query: str) -> str:
        """Process user voice query against contextual visual memory."""
        return self._memory.answer_query(query, ollama_client=self._ollama)

    def get_ui_overlay_state(self) -> Dict[str, Any]:
        """Return current UI overlay state for VisionRenderer HUD."""
        return dict(self._ui_state)
