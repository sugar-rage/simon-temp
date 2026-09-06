"""Health Monitor — aggregates system health and generates diagnostic reports.

Combines data from CapabilityRegistry, Watchdog, ModelRegistry, and
StateMachine to produce human-readable status reports (for TTS via the
``status`` voice command) and detailed machine-readable diagnostics.

Metrics:
- ``system.uptime_s`` — gauge tracking system uptime
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.capabilities.registry import CapabilityRegistry
from core.health.watchdog import Watchdog
from core.resources.model_registry import ModelRegistry
from core.state.state_machine import StateMachine, SystemState
from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.metrics.collector import SystemMetricsCollector

logger = logging.getLogger("simon.core.health")


class HealthMonitor:
    """Aggregates system health and produces diagnostic reports.

    Parameters
    ----------
    capabilities : CapabilityRegistry
        Runtime capability status.
    watchdog : Watchdog
        Thread heartbeat monitor.
    model_registry : ModelRegistry
        Loaded ML model inventory.
    state_machine : StateMachine
        System state.
    event_bus : EventBus
        For publishing health check events.
    start_time : float, optional
        System start timestamp (for uptime calc).
    """

    def __init__(
        self,
        capabilities: CapabilityRegistry,
        watchdog: Watchdog,
        model_registry: ModelRegistry,
        state_machine: StateMachine,
        event_bus: EventBus,
        start_time: Optional[float] = None,
    ) -> None:
        self._capabilities = capabilities
        self._watchdog = watchdog
        self._model_registry = model_registry
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._start_time = start_time or time.time()
        self._metrics = SystemMetricsCollector.get()

    def get_status_report(self) -> str:
        """Generate a TTS-friendly status report.

        Returns
        -------
        str
            Human-readable status (e.g. "SIMON is running. Camera active.
            I see 3 objects. GPS weak. Navigation idle. All systems operational.")
        """
        parts = []
        state = self._state_machine.state

        # System state
        if state == SystemState.READY:
            parts.append("SIMON is running")
        elif state == SystemState.DEGRADED:
            parts.append("SIMON is running in degraded mode")
        elif state == SystemState.NAVIGATING:
            parts.append("SIMON is navigating")
        elif state == SystemState.PAUSED:
            parts.append("SIMON is paused")
        else:
            parts.append(f"SIMON state is {state.name.lower()}")

        # Camera
        if self._capabilities.is_available("capability.camera"):
            parts.append("Camera active")
        else:
            parts.append("Camera unavailable")

        # Detection
        if self._capabilities.is_available("capability.detection"):
            parts.append("Object detection active")

        # GPS
        if self._capabilities.is_available("capability.gps"):
            gps_status = self._capabilities.get_status("capability.gps")
            if gps_status and gps_status.degraded:
                parts.append("GPS signal weak")
            else:
                parts.append("GPS active")
        else:
            parts.append("GPS unavailable")

        # Navigation state
        if state == SystemState.NAVIGATING:
            parts.append("Navigation in progress")
        else:
            parts.append("Navigation idle")

        # OCR
        if not self._capabilities.is_available("capability.ocr"):
            parts.append("Text reading unavailable")

        # Face recognition
        if not self._capabilities.is_available("capability.face_recognition"):
            parts.append("Face recognition unavailable")

        # Overall health
        degraded = self._capabilities.list_degraded()
        unavailable = self._capabilities.list_unavailable()
        if not degraded and not unavailable:
            parts.append("All systems operational")
        elif unavailable:
            parts.append(f"{len(unavailable)} systems unavailable")

        # Uptime
        uptime_s = time.time() - self._start_time
        self._metrics.set_gauge("system.uptime_s", uptime_s)
        if uptime_s > 3600:
            hours = int(uptime_s // 3600)
            mins = int((uptime_s % 3600) // 60)
            parts.append(f"Uptime {hours} hours {mins} minutes")
        elif uptime_s > 60:
            mins = int(uptime_s // 60)
            parts.append(f"Uptime {mins} minutes")

        return ". ".join(parts) + "."

    def get_detailed_report(self) -> dict[str, Any]:
        """Generate a detailed machine-readable health report.

        Returns
        -------
        dict
            Comprehensive health data.
        """
        uptime_s = time.time() - self._start_time
        self._metrics.set_gauge("system.uptime_s", uptime_s)

        return {
            "state": self._state_machine.state.name,
            "uptime_s": uptime_s,
            "capabilities": {
                "available": self._capabilities.list_available(),
                "degraded": self._capabilities.list_degraded(),
                "unavailable": self._capabilities.list_unavailable(),
            },
            "threads": self._watchdog.get_registered_threads(),
            "models": [
                {
                    "name": m.name,
                    "device": m.device,
                    "size_mb": m.size_mb,
                    "priority": m.priority,
                }
                for m in self._model_registry.list_models()
            ],
            "model_count": self._model_registry.model_count,
            "watchdog_running": self._watchdog.is_running,
        }

    def check_health(self) -> None:
        """Run a health check and publish results.

        Publishes a ``health.check`` event with the detailed report.
        """
        try:
            report = self.get_detailed_report()
            self._event_bus.publish(Event(
                topic=event_types.HEALTH_CHECK,
                priority=Priority.LOW,
                source="health_monitor",
                data=report,
            ))
        except Exception:
            logger.error("Health check failed", exc_info=True)
