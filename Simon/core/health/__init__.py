"""Health Monitor & Watchdog — subsystem health tracking and thread liveness.

Provides:
- HealthMonitor: periodic health checks across subsystems
- Watchdog: detects dead threads and triggers recovery
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority, SubsystemStatus

logger = logging.getLogger("simon.core.health")


@dataclass
class HealthStatus:
    """Health status of a single subsystem."""

    name: str
    status: SubsystemStatus = SubsystemStatus.UNKNOWN
    message: str = ""
    last_checked: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Periodic health checker for all subsystems.

    Parameters
    ----------
    event_bus : EventBus, optional
        For publishing health events.
    check_interval_s : float
        Interval between health check rounds.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        check_interval_s: float = 30.0,
    ) -> None:
        self._event_bus = event_bus
        self._interval = check_interval_s
        self._checks: dict[str, Callable[[], HealthStatus]] = {}
        self._statuses: dict[str, HealthStatus] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register(self, name: str, check_fn: Callable[[], HealthStatus]) -> None:
        """Register a health check function for a subsystem."""
        self._checks[name] = check_fn

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, name="HealthMonitor", daemon=True
        )
        self._thread.start()
        logger.info("Health monitor started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def check_all(self) -> dict[str, HealthStatus]:
        """Run all health checks immediately."""
        for name, check_fn in self._checks.items():
            try:
                status = check_fn()
                self._statuses[name] = status
            except Exception as e:
                self._statuses[name] = HealthStatus(
                    name=name, status=SubsystemStatus.ERROR, message=str(e),
                )
        return dict(self._statuses)

    def get_status(self, name: str) -> Optional[HealthStatus]:
        return self._statuses.get(name)

    @property
    def all_healthy(self) -> bool:
        return all(
            s.status in (SubsystemStatus.HEALTHY, SubsystemStatus.UNKNOWN)
            for s in self._statuses.values()
        )

    def _monitor_loop(self) -> None:
        while self._running:
            statuses = self.check_all()
            for name, status in statuses.items():
                if self._event_bus and status.status == SubsystemStatus.ERROR:
                    self._event_bus.publish_sync(Event(
                        topic=event_types.HEALTH_CHECK,
                        data={"subsystem": name, "status": status.status.name, "message": status.message},
                        priority=Priority.OBSTACLE,
                        source="health_monitor",
                    ))
            time.sleep(self._interval)


class Watchdog:
    """Thread liveness watchdog with heartbeats and auto-recovery.

    Parameters
    ----------
    event_bus : EventBus, optional
    check_interval_s : float
    timeout_s : float
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        check_interval_s: float = 5.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._event_bus = event_bus
        self._interval = check_interval_s
        self._timeout = timeout_s
        self._heartbeats: dict[str, float] = {}
        self._recovery_callbacks: dict[str, Callable[[], None]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_thread(self, name: str, recovery_fn: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            self._heartbeats[name] = time.time()
            if recovery_fn:
                self._recovery_callbacks[name] = recovery_fn

    def heartbeat(self, name: str) -> None:
        with self._lock:
            self._heartbeats[name] = time.time()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, name="Watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def check(self) -> list[str]:
        """Returns names of dead threads."""
        dead: list[str] = []
        now = time.time()
        with self._lock:
            for name, last_hb in self._heartbeats.items():
                if now - last_hb > self._timeout:
                    dead.append(name)
        return dead

    def _watch_loop(self) -> None:
        while self._running:
            dead = self.check()
            for name in dead:
                logger.error("Thread dead: %s", name)
                if self._event_bus:
                    self._event_bus.publish_sync(Event(
                        topic=event_types.HEALTH_THREAD_DEAD,
                        data={"thread_name": name},
                        priority=Priority.SAFETY_CRITICAL,
                        source="watchdog",
                    ))
                recovery = self._recovery_callbacks.get(name)
                if recovery:
                    try:
                        recovery()
                        with self._lock:
                            self._heartbeats[name] = time.time()
                    except Exception as e:
                        logger.error("Recovery failed for %s: %s", name, e)
            time.sleep(self._interval)
