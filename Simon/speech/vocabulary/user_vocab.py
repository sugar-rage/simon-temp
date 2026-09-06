"""
User Vocabulary — persistent user-defined terms.

Stores custom words the user has explicitly taught the system
(e.g. names of people, brand names, technical jargon).
Persisted across sessions via a simple JSON file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class UserVocab:
    """Manages user-defined vocabulary terms.

    Args:
        persist_path: Path to JSON file for persistence. None = in-memory only.
        max_entries:  Maximum stored terms.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_entries: int = 500,
    ):
        self._terms: List[str] = []
        self._max = max_entries
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def add(self, term: str) -> None:
        """Add a user-defined term."""
        normalized = term.strip()
        if not normalized:
            return
        lower_set = {t.lower() for t in self._terms}
        if normalized.lower() not in lower_set:
            self._terms.append(normalized)
            if len(self._terms) > self._max:
                self._terms.pop(0)
            self._save()

    def remove(self, term: str) -> bool:
        """Remove a term. Returns True if found."""
        normalized = term.strip().lower()
        for i, t in enumerate(self._terms):
            if t.lower() == normalized:
                self._terms.pop(i)
                self._save()
                return True
        return False

    def get_terms(self) -> List[str]:
        """Return all user-defined terms."""
        return list(self._terms)

    def clear(self) -> None:
        self._terms.clear()
        self._save()

    @property
    def size(self) -> int:
        return len(self._terms)

    def _load(self) -> None:
        """Load terms from disk."""
        if self._persist_path and self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._terms = data.get("terms", [])[:self._max]
                logger.debug(f"Loaded {len(self._terms)} user vocab terms")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load user vocab: {e}")

    def _save(self) -> None:
        """Persist terms to disk."""
        if self._persist_path:
            try:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                self._persist_path.write_text(
                    json.dumps({"terms": self._terms}, indent=2),
                    encoding="utf-8",
                )
            except OSError as e:
                logger.warning(f"Failed to save user vocab: {e}")
