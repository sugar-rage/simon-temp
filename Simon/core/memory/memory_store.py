"""Session and Persistent Memory — remembers what SIMON has seen and done.

SessionMemory: volatile, per-run memory (cleared on restart).
PersistentMemory: disk-backed memory using JSON (survives restarts).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("simon.core.memory")


@dataclass
class MemoryEntry:
    """A single memory entry.

    Attributes
    ----------
    key : str
        Unique entry key.
    value : Any
        Stored value.
    category : str
        Category tag (e.g. "detection", "face", "navigation").
    timestamp : float
        When the entry was created/updated.
    ttl_s : float
        Time-to-live in seconds (0 = never expire).
    """

    key: str
    value: Any
    category: str = "general"
    timestamp: float = field(default_factory=time.time)
    ttl_s: float = 0.0

    @property
    def is_expired(self) -> bool:
        if self.ttl_s <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl_s


class SessionMemory:
    """Volatile per-session memory — cleared on system restart.

    Stores detection history, recent announcements, encounter counts,
    and other runtime data.

    Parameters
    ----------
    max_entries : int
        Maximum stored entries before eviction.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        ttl_s: float = 0.0,
    ) -> None:
        """Store a value in session memory."""
        with self._lock:
            self._entries[key] = MemoryEntry(
                key=key, value=value, category=category, ttl_s=ttl_s
            )
            self._evict_expired()
            if len(self._entries) > self._max_entries:
                self._evict_oldest()

    def recall(self, key: str) -> Optional[Any]:
        """Recall a value from session memory."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._entries[key]
                return None
            return entry.value

    def recall_by_category(self, category: str) -> list[MemoryEntry]:
        """Return all non-expired entries in a category."""
        with self._lock:
            return [
                e for e in self._entries.values()
                if e.category == category and not e.is_expired
            ]

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter in memory. Returns the new value."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.is_expired:
                self._entries[key] = MemoryEntry(
                    key=key, value=amount, category="counter"
                )
                return amount
            entry.value = (entry.value or 0) + amount
            entry.timestamp = time.time()
            return entry.value

    def forget(self, key: str) -> bool:
        """Remove an entry. Returns True if it existed."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def _evict_expired(self) -> None:
        expired = [k for k, v in self._entries.items() if v.is_expired]
        for k in expired:
            del self._entries[k]

    def _evict_oldest(self) -> None:
        sorted_entries = sorted(
            self._entries.items(), key=lambda x: x[1].timestamp
        )
        to_remove = len(self._entries) - self._max_entries
        for key, _ in sorted_entries[:max(0, to_remove)]:
            del self._entries[key]


class PersistentMemory:
    """Disk-backed memory that survives system restarts.

    Stores data in a JSON file. Used for:
    - Known face encounter history
    - User preferences
    - Favorite locations
    - Long-term statistics

    Parameters
    ----------
    filepath : str
        Path to the JSON storage file.
    """

    def __init__(self, filepath: str = "data/persistent_memory.json") -> None:
        self._filepath = filepath
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load()

    def store(self, key: str, value: Any) -> None:
        """Store a value and persist to disk."""
        with self._lock:
            self._data[key] = {
                "value": value,
                "timestamp": time.time(),
            }
            self._save()

    def recall(self, key: str) -> Optional[Any]:
        """Recall a value from persistent memory."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            return entry.get("value")

    def forget(self, key: str) -> bool:
        """Remove a value and persist."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._save()
                return True
            return False

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._save()

    def _load(self) -> None:
        """Load from disk."""
        if not os.path.exists(self._filepath):
            return
        try:
            with open(self._filepath, "r") as f:
                self._data = json.load(f)
            logger.info("Loaded persistent memory: %d entries", len(self._data))
        except Exception as e:
            logger.warning("Failed to load persistent memory: %s", e)
            self._data = {}

    def _save(self) -> None:
        """Save to disk."""
        try:
            os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
            with open(self._filepath, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save persistent memory: %s", e)
