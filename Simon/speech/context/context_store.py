"""
Context Store — TTL-based slot storage for conversational memory.

Maintains named context slots (e.g. ``last_location``, ``last_action``,
``last_object``) with automatic expiration.  Each slot has a value, a
timestamp, and a configurable time-to-live.

Thread-safe: all mutations acquire a lock so the ListenPipeline and
SpeechManager can read/write safely from different threads.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextSlot:
    """A single named context value with metadata.

    Attributes:
        name:       Slot identifier (e.g. ``"last_location"``).
        value:      The stored value (typically a string).
        timestamp:  Monotonic time when the slot was last written.
        ttl_s:      Time-to-live in seconds.  ``0`` = never expires.
        source:     How this slot was populated (e.g. ``"stt"``, ``"user"``).
    """

    name: str
    value: Any
    timestamp: float = field(default_factory=time.monotonic)
    ttl_s: float = 120.0  # Default: 2 minutes
    source: str = "system"

    @property
    def is_expired(self) -> bool:
        """True if the slot has exceeded its TTL."""
        if self.ttl_s <= 0:
            return False
        return (time.monotonic() - self.timestamp) > self.ttl_s

    @property
    def age_s(self) -> float:
        """Seconds since the slot was last written."""
        return time.monotonic() - self.timestamp


class ContextStore:
    """Thread-safe, TTL-based context slot store.

    Args:
        default_ttl_s:  Default TTL for slots that don't specify one.
        max_slots:      Maximum number of slots (oldest evicted on overflow).
    """

    def __init__(self, default_ttl_s: float = 120.0, max_slots: int = 50):
        self._default_ttl = default_ttl_s
        self._max_slots = max_slots
        self._slots: Dict[str, ContextSlot] = {}
        self._lock = threading.Lock()

    def set(
        self,
        name: str,
        value: Any,
        ttl_s: Optional[float] = None,
        source: str = "system",
    ) -> None:
        """Set or update a context slot.

        Args:
            name:   Slot identifier.
            value:  Value to store.
            ttl_s:  Override TTL for this slot. ``None`` → default.
            source: Origin tag for debugging.
        """
        with self._lock:
            self._slots[name] = ContextSlot(
                name=name,
                value=value,
                timestamp=time.monotonic(),
                ttl_s=ttl_s if ttl_s is not None else self._default_ttl,
                source=source,
            )

            # Evict oldest if over capacity
            if len(self._slots) > self._max_slots:
                self._evict_oldest()

        logger.debug(f"Context slot set: {name}={value!r} (ttl={ttl_s or self._default_ttl}s)")

    def get(self, name: str, default: Any = None) -> Any:
        """Get a slot's value, or *default* if missing or expired.

        Expired slots are lazily removed on access.
        """
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                return default
            if slot.is_expired:
                del self._slots[name]
                logger.debug(f"Context slot expired: {name} (age={slot.age_s:.1f}s)")
                return default
            return slot.value

    def get_slot(self, name: str) -> Optional[ContextSlot]:
        """Get the full ``ContextSlot`` object (or ``None``)."""
        with self._lock:
            slot = self._slots.get(name)
            if slot is None or slot.is_expired:
                return None
            return slot

    def has(self, name: str) -> bool:
        """True if *name* exists and is not expired."""
        return self.get(name) is not None

    def remove(self, name: str) -> bool:
        """Remove a slot. Returns True if it existed."""
        with self._lock:
            return self._slots.pop(name, None) is not None

    def clear(self) -> None:
        """Remove all slots."""
        with self._lock:
            self._slots.clear()

    def purge_expired(self) -> int:
        """Remove all expired slots. Returns count of removed slots."""
        with self._lock:
            expired = [n for n, s in self._slots.items() if s.is_expired]
            for n in expired:
                del self._slots[n]
        if expired:
            logger.debug(f"Purged {len(expired)} expired context slots")
        return len(expired)

    def all_active(self) -> Dict[str, Any]:
        """Return a snapshot of all non-expired slot values."""
        with self._lock:
            return {
                n: s.value for n, s in self._slots.items() if not s.is_expired
            }

    @property
    def size(self) -> int:
        """Number of slots (including possibly-expired ones)."""
        return len(self._slots)

    def _evict_oldest(self) -> None:
        """Remove the oldest slot (by timestamp). Caller must hold lock."""
        if not self._slots:
            return
        oldest = min(self._slots.values(), key=lambda s: s.timestamp)
        del self._slots[oldest.name]
        logger.debug(f"Evicted oldest context slot: {oldest.name}")
