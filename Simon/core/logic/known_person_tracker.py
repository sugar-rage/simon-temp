"""Known Person Tracker - tracks presence and re-entry of recognized people.

Maintains presence state per known person (by name/identity, not track_id) to prevent
repetitive arrival announcements while the person remains present or returns
within the re-announce timeout (default 5 minutes / 300 seconds).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("simon.core.logic.known_person")
KNOWN_PERSON_REANNOUNCE_TIMEOUT_S: float = 300.0


@dataclass
class KnownPersonState:
    name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_announced: float = field(default_factory=time.time)
    announcement_count: int = 1


class KnownPersonTracker:
    def __init__(self, reannounce_timeout_s: float = KNOWN_PERSON_REANNOUNCE_TIMEOUT_S, max_entries: int = 500) -> None:
        self._timeout = float(reannounce_timeout_s)
        self._max_entries = max_entries
        self._persons: dict[str, KnownPersonState] = {}

    @property
    def reannounce_timeout_s(self) -> float:
        return self._timeout

    @reannounce_timeout_s.setter
    def reannounce_timeout_s(self, value: float) -> None:
        self._timeout = float(value)

    def process_detection(self, name: str, now: Optional[float] = None) -> bool:
        if not name or name == "Unknown":
            return False
        if now is None:
            now = time.time()
        logger.info("[KNOWN_PERSON] detected name=%s", name)
        if name not in self._persons:
            logger.info("[KNOWN_PERSON] first arrival name=%s -> announcing", name)
            self._persons[name] = KnownPersonState(name=name, first_seen=now, last_seen=now, last_announced=now, announcement_count=1)
            self._evict_stale(now)
            return True
        state = self._persons[name]
        absence = now - state.last_seen
        if absence >= self._timeout:
            logger.info("[KNOWN_PERSON] re-entry name=%s absence=%.1fs -> announcing", name, absence)
            state.last_seen = now
            state.last_announced = now
            state.announcement_count += 1
            return True
        else:
            if absence <= 3.0:
                logger.info("[KNOWN_PERSON] already present name=%s -> suppressing announcement", name)
            else:
                logger.info("[KNOWN_PERSON] reappeared name=%s absence=%.1fs -> suppressing announcement", name, absence)
            state.last_seen = now
            return False

    def get_state(self, name: str) -> Optional[KnownPersonState]:
        return self._persons.get(name)

    def clear(self) -> None:
        self._persons.clear()

    def _evict_stale(self, now: float) -> None:
        if len(self._persons) <= self._max_entries:
            return
        stale_threshold = 2.0 * self._timeout
        to_delete = [n for n, s in self._persons.items() if (now - s.last_seen) > stale_threshold]
        for n in to_delete:
            del self._persons[n]
        if len(self._persons) > self._max_entries:
            sorted_names = sorted(self._persons.keys(), key=lambda n: self._persons[n].last_seen)
            for n in sorted_names[: len(self._persons) - self._max_entries]:
                del self._persons[n]
