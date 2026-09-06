"""
Command History — frequency-weighted command usage tracking.

Tracks which commands the user issues most often and how recently,
enabling:
- Priority boosting for frequently used commands in fuzzy matching
- Personalized help suggestions
- Usage analytics for the dashboard
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CommandRecord:
    """Usage record for a single command action."""

    action: str
    count: int = 0
    last_used: float = 0.0  # Unix timestamp
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    @property
    def recency_weight(self) -> float:
        """Higher for recently used commands (exponential decay)."""
        age_hours = (time.time() - self.last_used) / 3600
        return 2.0 ** (-age_hours / 24)  # Half-life = 24 hours


class CommandHistory:
    """Tracks command usage frequency and recency.

    Args:
        persist_path: Path for JSON persistence. None = in-memory only.
        max_actions:  Maximum unique actions to track.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_actions: int = 200,
    ):
        self._records: Dict[str, CommandRecord] = {}
        self._max = max_actions
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def record(self, action: str, success: bool = True) -> None:
        """Record a command execution.

        Args:
            action:  Action name.
            success: Whether the command executed successfully.
        """
        if action not in self._records:
            if len(self._records) >= self._max:
                self._evict_least_used()
            self._records[action] = CommandRecord(action=action)

        rec = self._records[action]
        rec.count += 1
        rec.last_used = time.time()
        if success:
            rec.success_count += 1
        else:
            rec.failure_count += 1

    def get_frequency_boost(self, action: str) -> float:
        """Get a frequency/recency boost score for an action.

        Returns a value ≥ 0 where higher = more frequently used recently.
        Used by the IntentParser for fuzzy match tiebreaking.
        """
        rec = self._records.get(action)
        if rec is None:
            return 0.0
        return rec.count * rec.recency_weight

    def get_top_commands(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the N most-used commands as (action, count) pairs."""
        sorted_recs = sorted(
            self._records.values(),
            key=lambda r: r.count * r.recency_weight,
            reverse=True,
        )
        return [(r.action, r.count) for r in sorted_recs[:n]]

    def get_record(self, action: str) -> Optional[CommandRecord]:
        return self._records.get(action)

    @property
    def size(self) -> int:
        return len(self._records)

    def save(self) -> None:
        """Persist to disk."""
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: {
                    "action": r.action,
                    "count": r.count,
                    "last_used": r.last_used,
                    "success_count": r.success_count,
                    "failure_count": r.failure_count,
                }
                for k, r in self._records.items()
            }
            self._persist_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning(f"Failed to save command history: {e}")

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self._records[k] = CommandRecord(**v)
            logger.debug(f"Loaded {len(self._records)} command history records")
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(f"Failed to load command history: {e}")

    def _evict_least_used(self) -> None:
        """Evict the least recently/frequently used action."""
        if not self._records:
            return
        worst = min(
            self._records.values(),
            key=lambda r: r.count * r.recency_weight,
        )
        del self._records[worst.action]
