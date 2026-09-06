"""Announcement Tracker — prevents repetitive TTS announcements.

Tracks what was announced and when, applying configurable cooldowns
to avoid annoying the user with repeated announcements of the same
object/hazard/text.
"""

from __future__ import annotations

import time
import threading
from typing import Optional


class AnnouncementTracker:
    """Tracks announced items and enforces cooldown periods.

    Parameters
    ----------
    default_cooldown_s : float
        Default cooldown between repeated announcements (seconds).
    hazard_cooldown_s : float
        Shorter cooldown for hazard re-announcements.
    face_cooldown_s : float
        Cooldown for face re-announcements.
    max_entries : int
        Maximum tracked entries (prevents memory leaks).
    """

    def __init__(
        self,
        default_cooldown_s: float = 15.0,
        hazard_cooldown_s: float = 3.0,
        face_cooldown_s: float = 30.0,
        max_entries: int = 500,
    ) -> None:
        self._default_cooldown = default_cooldown_s
        self._hazard_cooldown = hazard_cooldown_s
        self._face_cooldown = face_cooldown_s
        self._max_entries = max_entries
        self._entries: dict[str, float] = {}
        self._lock = threading.Lock()

    def was_announced(
        self, key: str, cooldown_s: Optional[float] = None
    ) -> bool:
        """Check if an item was announced within its cooldown period.

        Parameters
        ----------
        key : str
            Unique announcement key (e.g. "car:left", "face:Alice").
        cooldown_s : float, optional
            Override cooldown for this check.
        """
        with self._lock:
            last = self._entries.get(key)
            if last is None:
                return False

            cooldown = cooldown_s or self._get_cooldown(key)
            return (time.time() - last) < cooldown

    def mark_announced(self, key: str) -> None:
        """Mark an item as announced at the current time."""
        with self._lock:
            self._entries[key] = time.time()
            self._evict_if_needed()

    def clear(self) -> None:
        """Clear all tracked announcements."""
        with self._lock:
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def _get_cooldown(self, key: str) -> float:
        """Determine the appropriate cooldown for a key."""
        if key.startswith("hazard:"):
            return self._hazard_cooldown
        elif key.startswith("face:"):
            return self._face_cooldown
        return self._default_cooldown

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if over capacity."""
        if len(self._entries) <= self._max_entries:
            return

        # Remove oldest entries
        sorted_entries = sorted(self._entries.items(), key=lambda x: x[1])
        to_remove = len(self._entries) - self._max_entries
        for key, _ in sorted_entries[:to_remove]:
            del self._entries[key]
