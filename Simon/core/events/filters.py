"""Event filters — priority filtering, topic matching, rate limiting."""

from __future__ import annotations

import time
import threading
from typing import Optional

from core.models.events import Event
from core.models.enums import Priority


class PriorityFilter:
    """Filter events by priority threshold.

    Events with priority > max_priority (i.e. lower urgency) are rejected.

    Parameters
    ----------
    max_priority : int
        Maximum priority value to accept (higher value = lower urgency).
        Defaults to ``Priority.DEBUG`` (accept everything).
    """

    def __init__(self, max_priority: int = Priority.DEBUG) -> None:
        self._max_priority = max_priority

    def accepts(self, event: Event) -> bool:
        """Return True if the event's priority passes the filter."""
        return event.priority <= self._max_priority


class TopicFilter:
    """Filter events by topic pattern matching.

    Supports exact match and wildcard (``"vision.*"`` matches all vision events).

    Parameters
    ----------
    patterns : list[str]
        Topic patterns to accept.  ``"*"`` matches everything.
        ``"vision.*"`` matches ``"vision.detection"``, ``"vision.face"``, etc.
    """

    def __init__(self, patterns: Optional[list[str]] = None) -> None:
        self._patterns = patterns or ["*"]

    def accepts(self, event: Event) -> bool:
        """Return True if the event's topic matches any pattern."""
        for pattern in self._patterns:
            if pattern == "*":
                return True
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event.topic.startswith(prefix):
                    return True
            elif pattern == event.topic:
                return True
        return False


class RateLimiter:
    """Rate-limit events per topic to prevent event storms.

    Thread-safe: uses a lock to protect the timing dictionary.

    Parameters
    ----------
    min_interval_s : float
        Minimum seconds between events on the same topic.
    """

    def __init__(self, min_interval_s: float = 0.1) -> None:
        self._min_interval = min_interval_s
        self._last_emit: dict[str, float] = {}
        self._lock = threading.Lock()

    def allows(self, event: Event) -> bool:
        """Return True if enough time has passed since the last event on this topic.

        Safety-critical events always pass.
        """
        if event.is_safety_critical:
            return True

        now = time.time()
        with self._lock:
            last = self._last_emit.get(event.topic, 0.0)
            if now - last < self._min_interval:
                return False
            self._last_emit[event.topic] = now
            return True

    def reset(self, topic: Optional[str] = None) -> None:
        """Reset rate limiting state."""
        with self._lock:
            if topic:
                self._last_emit.pop(topic, None)
            else:
                self._last_emit.clear()


class CompositeFilter:
    """Chains multiple filters together — event must pass ALL filters."""

    def __init__(
        self,
        priority_filter: Optional[PriorityFilter] = None,
        topic_filter: Optional[TopicFilter] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._filters = []
        if priority_filter:
            self._filters.append(priority_filter.accepts)
        if topic_filter:
            self._filters.append(topic_filter.accepts)
        if rate_limiter:
            self._filters.append(rate_limiter.allows)

    def accepts(self, event: Event) -> bool:
        """Return True only if the event passes all constituent filters."""
        return all(f(event) for f in self._filters)
