"""
Session Vocabulary — session-scoped dynamic terms.

Accumulates vocabulary terms discovered during a single session
(e.g. OCR-detected text, object labels from computer vision,
sign text read by the camera).  Cleared on session end.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class SessionVocab:
    """Manages session-scoped vocabulary terms.

    These terms are injected into the Whisper ``initial_prompt`` and
    are cleared when the session ends.

    Args:
        max_entries: Maximum terms per session.
    """

    def __init__(self, max_entries: int = 100):
        self._terms: List[str] = []
        self._max = max_entries

    def add(self, term: str) -> None:
        """Add a session-discovered term (e.g. from OCR)."""
        normalized = term.strip()
        if not normalized:
            return
        lower_set = {t.lower() for t in self._terms}
        if normalized.lower() not in lower_set:
            self._terms.append(normalized)
            if len(self._terms) > self._max:
                self._terms.pop(0)

    def add_many(self, terms: List[str]) -> None:
        """Add multiple terms at once."""
        for t in terms:
            self.add(t)

    def get_terms(self) -> List[str]:
        """Return all session terms."""
        return list(self._terms)

    def clear(self) -> None:
        """Clear all session terms (call on session end)."""
        self._terms.clear()

    @property
    def size(self) -> int:
        return len(self._terms)
