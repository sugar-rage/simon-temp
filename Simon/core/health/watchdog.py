"""Watchdog — thread heartbeat monitoring and deadlock detection.

Each subsystem thread periodically calls ``heartbeat(name)`` to
signal liveness.  The watchdog runs a background thread that checks
heartbeat timestamps.  If a thread's heartbeat is stale beyond the
timeout, a ``health.thread_dead`` event is published.

Design decision: the watchdog is a daemon thread so it does not
block shutdown.  Thread death detection is best-effort — the
watchdog can only detect stalled threads, not deadlocks between
threads that are blocked on each other.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority

logger = logging.getLogger("simon.core.health")


class Watchdog:
    """Monitors thread heartbeats and detects stalled threads.

    Parameters
    ----------
    event_bus : EventBus
        For publishing ``health.thread_dead`` events.
    interval_s : float
        How often to check heartbeats (seconds).
    timeout_s : float
        How long a heartbeat can be stale before alerting (seconds).
    """

    def __init__(
        self,
        event_bus: EventBus,
        interval_s: float = 5.0,
        timeout_s: float = 15.0,
    ) -> None:
        self._event_bus = event_bus
        self._interval = interval_s
        self._timeout = timeout_s
        self._heartbeats: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_thread(self, name: str) -> None:
        """Register a thread for monitoring.

        Sets the initial heartbeat timestamp to now.

        Parameters
        ----------
        name : str
            Human-readable thread identifier.
        """
        with self._lock:
            self._heartbeats[name] = time.time()
        logger.debug("Registered thread %r for watchdog monitoring", name)

    def unregister_thread(self, name: str) -> None:
        """Remove a thread from monitoring."""
        with self._lock:
            self._heartbeats.pop(name, None)

    def heartbeat(self, name: str) -> None:
        """Record a heartbeat from a thread.

        Parameters
        ----------
        name : str
            Thread identifier (must match ``register_thread`` name).
        """
        with self._lock:
            if name in self._heartbeats:
                self._heartbeats[name] = time.time()

    def start(self) -> None:
        """Start the watchdog background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="watchdog", daemon=True,
        )
        self._thread.start()
        logger.info("Watchdog started (interval=%.1fs, timeout=%.1fs)",
                     self._interval, self._timeout)

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None
        logger.info("Watchdog stopped")

    @property
    def is_running(self) -> bool:
        """True if the watchdog thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_registered_threads(self) -> dict[str, float]:
        """Return snapshot of all registered threads and their last heartbeat."""
        with self._lock:
            return dict(self._heartbeats)

    # ── Internal ─────────────────────────────────────────────────────

    def _run(self) -> None:
        """Watchdog main loop."""
        while not self._stop_event.wait(self._interval):
            self._check_heartbeats()

    def _check_heartbeats(self) -> None:
        """Check all heartbeats for staleness."""
        now = time.time()
        with self._lock:
            stale = {
                name: ts for name, ts in self._heartbeats.items()
                if (now - ts) > self._timeout
            }

        for name, last_ts in stale.items():
            age = now - last_ts
            logger.error(
                "Thread %r heartbeat stale (%.1fs ago, timeout=%.1fs)",
                name, age, self._timeout,
            )
            try:
                self._event_bus.publish(Event(
                    topic=event_types.HEALTH_THREAD_DEAD,
                    priority=Priority.SAFETY_CRITICAL,
                    source="watchdog",
                    data={
                        "thread_name": name,
                        "last_heartbeat": last_ts,
                        "stale_seconds": age,
                    },
                    is_safety_critical=True,
                ))
            except Exception:
                logger.error("Failed to publish thread_dead event", exc_info=True)
