"""Visual Event Tracker - deduplicates OCR and visual events across video frames.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger("simon.core.visual_context.tracker")


@dataclass
class TrackedEventState:
    """State of an active or recent visual event."""
    key: str
    raw_text: str
    classification: str
    first_seen: float
    last_seen: float
    tts_started: bool = False
    tts_completed: bool = False
    tts_time: float = 0.0
    ollama_dispatched: bool = False
    ollama_completed: bool = False
    ollama_time: float = 0.0
    occurrence_count: int = 1


class VisualEventTracker:
    """Tracks visual event occurrences and manages deduplication / cooldowns."""

    def __init__(self, absence_reset_s: float = 10.0, max_entries: int = 200):
        self._absence_reset_s = absence_reset_s
        self._max_entries = max_entries
        self._events: Dict[str, TrackedEventState] = {}

    def should_process(self, key: str, now: Optional[float] = None) -> bool:
        """Check if an event key should trigger new TTS / Ollama processing.

        Returns True if:
        1. Event has never been seen before, or
        2. Event was absent for >= absence_reset_s (disappeared and returned).
        """
        if not key:
            return False
        if now is None:
            now = time.time()

        if key not in self._events:
            return True

        state = self._events[key]
        absence = now - state.last_seen
        if absence >= self._absence_reset_s:
            # Re-entry after absence
            return True

        # Continuously present -> only process if TTS hasn't started yet
        return not state.tts_started

    def register_seen(self, key: str, raw_text: str, classification: str, now: Optional[float] = None) -> TrackedEventState:
        """Record that an event is currently visible in the frame."""
        if now is None:
            now = time.time()

        if key not in self._events:
            state = TrackedEventState(
                key=key,
                raw_text=raw_text,
                classification=classification,
                first_seen=now,
                last_seen=now,
            )
            self._events[key] = state
            self._evict_stale(now)
            return state

        state = self._events[key]
        absence = now - state.last_seen
        if absence >= self._absence_reset_s:
            # Re-entry reset
            state.first_seen = now
            state.tts_started = False
            state.tts_completed = False
            state.ollama_dispatched = False
            state.ollama_completed = False
            state.occurrence_count += 1

        state.last_seen = now
        return state

    def mark_tts_started(self, key: str, now: Optional[float] = None) -> None:
        """Mark that TTS has begun speaking for this event."""
        if now is None:
            now = time.time()
        if key in self._events:
            self._events[key].tts_started = True
            self._events[key].tts_time = now

    def mark_tts_completed(self, key: str, now: Optional[float] = None) -> None:
        """Mark that TTS has fully finished for this event."""
        if now is None:
            now = time.time()
        if key in self._events:
            self._events[key].tts_completed = True

    def mark_ollama_dispatched(self, key: str) -> None:
        """Mark that Ollama processing has been triggered."""
        if key in self._events:
            self._events[key].ollama_dispatched = True

    def mark_ollama_completed(self, key: str, now: Optional[float] = None) -> None:
        """Mark that Ollama processing has completed."""
        if now is None:
            now = time.time()
        if key in self._events:
            self._events[key].ollama_completed = True
            self._events[key].ollama_time = now

    def get_event(self, key: str) -> Optional[TrackedEventState]:
        return self._events.get(key)

    def clear(self) -> None:
        self._events.clear()

    def _evict_stale(self, now: float) -> None:
        if len(self._events) <= self._max_entries:
            return
        stale_threshold = 300.0
        to_delete = [k for k, v in self._events.items() if (now - v.last_seen) > stale_threshold]
        for k in to_delete:
            del self._events[k]
