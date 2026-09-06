"""Contextual Visual Memory - stores structured visual intelligence for querying.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from core.visual_context.ollama_client import OllamaClient


@dataclass
class VisualMemoryEntry:
    """A structured record of visual observation and extracted intelligence."""
    timestamp: float
    raw_ocr: List[str]
    detected_objects: List[str]
    faces: List[str]
    classification: str
    spoken_response: Optional[str]
    direction: Optional[str]
    distance: Optional[str]
    event_type: Optional[str] = None
    place_name: Optional[str] = None
    category: Optional[str] = None
    recommended_action: Optional[str] = None
    summary: str = ""
    raw_json: Dict[str, Any] = field(default_factory=dict)


class ContextualVisualMemory:
    """Thread-safe memory store for visual context and user query answering."""

    def __init__(self, max_entries: int = 100):
        self._max_entries = max_entries
        self._entries: List[VisualMemoryEntry] = []
        self._lock = threading.Lock()

    def add_entry(self, entry: VisualMemoryEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def get_recent_entries(self, max_age_s: float = 300.0) -> List[VisualMemoryEntry]:
        """Return all entries recorded within max_age_s."""
        now = time.time()
        with self._lock:
            return [e for e in self._entries if (now - e.timestamp) <= max_age_s]

    def get_context_summary(self, max_age_s: float = 300.0) -> str:
        """Format recent memory into a clean text summary for Ollama prompting."""
        recent = self.get_recent_entries(max_age_s=max_age_s)
        if not recent:
            return "No recent visual memory recorded."

        lines = []
        for e in recent:
            parts = []
            if e.place_name:
                parts.append(f"Place: {e.place_name}")
            if e.event_type:
                parts.append(f"Event: {e.event_type}")
            if e.raw_ocr:
                parts.append(f"Sign: '{', '.join(e.raw_ocr)}'")
            if e.detected_objects:
                parts.append(f"Objects: {', '.join(e.detected_objects)}")
            if e.faces:
                parts.append(f"People: {', '.join(e.faces)}")
            if e.direction:
                parts.append(f"Direction: {e.direction}")
            if e.summary:
                parts.append(f"Summary: {e.summary}")
            lines.append(" - " + " | ".join(parts))
        return "\n".join(lines)

    def answer_query(self, query: str, ollama_client: Optional[OllamaClient] = None) -> str:
        """Answer a voice query using recent visual memory and optional Ollama intelligence."""
        context_str = self.get_context_summary(max_age_s=300.0)
        if ollama_client:
            return ollama_client.answer_query_sync(query, context_str)
        
        from core.visual_context.ollama_client import OllamaClient as FallbackClient
        return FallbackClient(enabled=False)._fallback_answer_query(query, context_str)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
