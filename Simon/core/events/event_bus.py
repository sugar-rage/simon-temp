"""Central Event Bus — pub/sub dispatcher for inter-subsystem communication.

This is the backbone of SIMON's architecture.  All subsystems communicate
exclusively through the Event Bus — no direct imports between subsystems.

See implementation_plan.md Section 3.1 for the full design.
See TDR-001 for the decision rationale.

Features:
- Topic-based pub/sub with wildcard matching (``"vision.*"``)
- Priority-based dispatch ordering
- Rate limiting to prevent event storms
- Safety-critical event bypass (synchronous, immediate dispatch)
- Correlation ID propagation for tracing
- Thread-safe subscriber management

Thread-safety:
- Subscriber registry is protected by ``threading.Lock``
- Event dispatch is serialized per subscriber (no concurrent handler calls)
- Safety-critical events are dispatched synchronously on the publisher's thread
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

from core.models.events import Event
from core.models.enums import Priority
from core.events.filters import RateLimiter
from core.metrics.collector import SystemMetricsCollector

logger = logging.getLogger("simon.core.events")

# Type alias for event handlers
EventHandler = Callable[[Event], None]


class _Subscription:
    """Internal subscription record."""

    __slots__ = ("handler", "topic_pattern", "priority_filter", "source")

    def __init__(
        self,
        handler: EventHandler,
        topic_pattern: str,
        priority_filter: int,
        source: str,
    ) -> None:
        self.handler = handler
        self.topic_pattern = topic_pattern
        self.priority_filter = priority_filter
        self.source = source

    def matches_topic(self, topic: str) -> bool:
        """Check if a topic matches this subscription's pattern."""
        if self.topic_pattern == "*":
            return True
        if self.topic_pattern.endswith(".*"):
            prefix = self.topic_pattern[:-2]
            return topic.startswith(prefix)
        return self.topic_pattern == topic

    def matches_priority(self, priority: int) -> bool:
        """Check if an event's priority passes this subscription's filter."""
        return priority <= self.priority_filter


class EventBus:
    """Central pub/sub event dispatcher.

    Parameters
    ----------
    rate_limit_s : float
        Minimum interval between events on the same topic (0 to disable).
    max_queue_size : int
        Maximum pending events in the dispatch queue.
    enable_metrics : bool
        Whether to collect dispatch metrics.
    """

    def __init__(
        self,
        rate_limit_s: float = 0.0,
        max_queue_size: int = 10_000,
        enable_metrics: bool = True,
    ) -> None:
        self._subscriptions: list[_Subscription] = []
        self._sub_lock = threading.Lock()

        self._rate_limiter: Optional[RateLimiter] = None
        if rate_limit_s > 0:
            self._rate_limiter = RateLimiter(min_interval_s=rate_limit_s)

        self._queue: queue.PriorityQueue[tuple[int, float, Event]] = (
            queue.PriorityQueue(maxsize=max_queue_size)
        )

        self._running = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self._enable_metrics = enable_metrics

    # ── Subscription ─────────────────────────────────────────────────

    def subscribe(
        self,
        topic_pattern: str,
        handler: EventHandler,
        *,
        priority_filter: int = Priority.DEBUG,
        source: str = "",
    ) -> None:
        """Subscribe a handler to events matching a topic pattern.

        Parameters
        ----------
        topic_pattern : str
            Topic to subscribe to.  Supports wildcards:
            ``"vision.*"`` matches all vision events.
            ``"*"`` matches everything.
        handler : callable
            Function called with ``Event`` when a matching event is published.
        priority_filter : int
            Only deliver events with priority ≤ this value.
            Default: accept all (``Priority.DEBUG``).
        source : str
            Identifier for this subscriber (for debugging).
        """
        sub = _Subscription(handler, topic_pattern, priority_filter, source)
        with self._sub_lock:
            self._subscriptions.append(sub)
        logger.debug(
            "Subscribed: %s → %s (priority ≤ %d)",
            source or "anonymous",
            topic_pattern,
            priority_filter,
        )

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove all subscriptions for a given handler."""
        with self._sub_lock:
            self._subscriptions = [
                s for s in self._subscriptions if s.handler is not handler
            ]

    # ── Publishing ───────────────────────────────────────────────────

    def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers.

        - **Safety-critical events** (``event.is_safety_critical``) are
          dispatched synchronously on the caller's thread for zero latency.
        - **Normal events** are queued for async dispatch by the dispatch thread.
        - Rate-limited events are silently dropped.
        """
        # Rate limiting (safety events always pass)
        if self._rate_limiter and not self._rate_limiter.allows(event):
            return

        if self._enable_metrics:
            SystemMetricsCollector.get().increment("core.event_count")

        if event.is_safety_critical:
            # Synchronous dispatch — zero latency for safety events
            self._dispatch_event(event)
        else:
            # Async dispatch via queue
            try:
                self._queue.put_nowait(
                    (event.priority, event.timestamp, event)
                )
            except queue.Full:
                logger.warning(
                    "Event queue full, dropping event: %s", event.topic
                )

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (bypasses the queue).

        Use sparingly — this blocks the caller until all handlers complete.
        Rate limiting is still applied.
        """
        if self._rate_limiter and not self._rate_limiter.allows(event):
            return
        self._dispatch_event(event)

    # ── Dispatch ─────────────────────────────────────────────────────

    def _dispatch_event(self, event: Event) -> None:
        """Dispatch an event to all matching subscribers."""
        with self._sub_lock:
            subscribers = list(self._subscriptions)

        start = time.perf_counter()

        for sub in subscribers:
            if sub.matches_topic(event.topic) and sub.matches_priority(
                event.priority
            ):
                try:
                    sub.handler(event)
                except Exception:
                    logger.exception(
                        "Handler error in %s for event %s",
                        sub.source or "unknown",
                        event.topic,
                    )

        if self._enable_metrics:
            elapsed_ms = (time.perf_counter() - start) * 1000
            SystemMetricsCollector.get().record_time(
                "core.event_latency_ms", elapsed_ms
            )

    def _dispatch_loop(self) -> None:
        """Background dispatch loop — processes queued events."""
        logger.info("Event bus dispatch thread started")
        while self._running:
            try:
                _, _, event = self._queue.get(timeout=0.1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in dispatch loop")
        logger.info("Event bus dispatch thread stopped")

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background dispatch thread."""
        if self._running:
            return
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventBus-Dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()
        logger.info("Event bus started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the dispatch thread and drain remaining events."""
        was_running = self._running
        self._running = False
        if was_running and self._dispatch_thread:
            self._dispatch_thread.join(timeout=timeout)
        # Drain remaining events (works even if bus was never started)
        drained = 0
        while not self._queue.empty():
            try:
                _, _, event = self._queue.get_nowait()
                self._dispatch_event(event)
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.info("Drained %d events during shutdown", drained)
        if was_running:
            logger.info("Event bus stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def subscriber_count(self) -> int:
        with self._sub_lock:
            return len(self._subscriptions)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def clear(self) -> None:
        """Remove all subscribers and drain queue (for testing)."""
        with self._sub_lock:
            self._subscriptions.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
