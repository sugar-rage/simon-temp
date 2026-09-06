"""
Location Vocabulary — location-specific terms for STT biasing.

Stores location names, landmarks, and place names that the user
frequently navigates to.  These are injected into the Whisper
``initial_prompt`` to bias recognition toward expected location names.
"""

from __future__ import annotations

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class LocationVocab:
    """Manages location-specific vocabulary terms.

    Args:
        initial_locations: Pre-seeded location names.
        max_entries:       Maximum stored locations (oldest evicted on overflow).
    """

    def __init__(
        self,
        initial_locations: List[str] | None = None,
        max_entries: int = 200,
    ):
        self._locations: List[str] = list(initial_locations or [])
        self._max = max_entries

    def add(self, location: str) -> None:
        """Add a location name. Deduplicates and caps at max_entries."""
        normalized = location.strip()
        if not normalized:
            return
        # Move to end if already present (most-recently-used ordering)
        lower_set = {l.lower() for l in self._locations}
        if normalized.lower() not in lower_set:
            self._locations.append(normalized)
            if len(self._locations) > self._max:
                self._locations.pop(0)  # Evict oldest

    def remove(self, location: str) -> bool:
        """Remove a location. Returns True if found."""
        normalized = location.strip().lower()
        for i, l in enumerate(self._locations):
            if l.lower() == normalized:
                self._locations.pop(i)
                return True
        return False

    def get_terms(self) -> List[str]:
        """Return all location terms."""
        return list(self._locations)

    def clear(self) -> None:
        self._locations.clear()

    @property
    def size(self) -> int:
        return len(self._locations)
