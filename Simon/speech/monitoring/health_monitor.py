"""
Health Monitor — component health checking for the SIMON speech subsystem.

Periodically checks the status of all major components and reports
their health.  Used by the dashboard and for automated error detection.

Each component registers a health check function that returns a
HealthStatus.  The monitor runs all checks and aggregates results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthState(Enum):
    """Health status of a component."""

    HEALTHY = auto()
    """Component is operating normally."""

    DEGRADED = auto()
    """Component is working but with reduced performance."""

    UNHEALTHY = auto()
    """Component has failed or is not responding."""

    UNKNOWN = auto()
    """Component health has not been checked."""


@dataclass
class HealthStatus:
    """Health check result for a single component.

    Attributes:
        component:  Component name (e.g. "stt", "vad", "tts").
        state:      Current health state.
        message:    Human-readable status message.
        latency_ms: Time taken for the health check.
        last_check: Monotonic time of last check.
        details:    Optional dict of component-specific metrics.
    """

    component: str
    state: HealthState = HealthState.UNKNOWN
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.monotonic)
    details: Dict = field(default_factory=dict)


class HealthMonitor:
    """Aggregates health checks from all speech subsystem components.

    Args:
        check_interval_s: Minimum seconds between consecutive full checks.
    """

    def __init__(self, check_interval_s: float = 30.0):
        self._checks: Dict[str, Callable[[], HealthStatus]] = {}
        self._results: Dict[str, HealthStatus] = {}
        self._interval = check_interval_s
        self._last_full_check: float = 0.0

    def register(self, component: str, check_fn: Callable[[], HealthStatus]) -> None:
        """Register a health check function for a component.

        Args:
            component: Component name.
            check_fn:  Callable that returns a HealthStatus.
        """
        self._checks[component] = check_fn
        self._results[component] = HealthStatus(
            component=component, state=HealthState.UNKNOWN
        )
        logger.debug(f"Health check registered: {component}")

    def check(self, component: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Run health checks.

        Args:
            component: If specified, check only this component.
                       If None, check all components.

        Returns:
            Dict of component name → HealthStatus.
        """
        if component:
            if component in self._checks:
                self._run_check(component)
            return {component: self._results.get(
                component,
                HealthStatus(component=component, state=HealthState.UNKNOWN),
            )}

        # Full check (throttled)
        now = time.monotonic()
        if (now - self._last_full_check) < self._interval:
            return dict(self._results)

        for name in self._checks:
            self._run_check(name)

        self._last_full_check = now
        return dict(self._results)

    def get_status(self, component: str) -> HealthStatus:
        """Get the last known status of a component."""
        return self._results.get(
            component,
            HealthStatus(component=component, state=HealthState.UNKNOWN),
        )

    @property
    def all_healthy(self) -> bool:
        """True if all registered components are healthy."""
        if not self._results:
            return True
        return all(
            s.state == HealthState.HEALTHY for s in self._results.values()
        )

    @property
    def unhealthy_components(self) -> List[str]:
        """List of component names that are unhealthy."""
        return [
            name for name, status in self._results.items()
            if status.state == HealthState.UNHEALTHY
        ]

    def get_summary(self) -> Dict[str, str]:
        """Get a human-readable summary of all component states."""
        return {
            name: f"{status.state.name}: {status.message}"
            for name, status in self._results.items()
        }

    def _run_check(self, component: str) -> None:
        """Execute a single health check."""
        check_fn = self._checks.get(component)
        if not check_fn:
            return

        start = time.monotonic()
        try:
            status = check_fn()
            status.latency_ms = (time.monotonic() - start) * 1000
            status.last_check = time.monotonic()
            self._results[component] = status
        except Exception as e:
            self._results[component] = HealthStatus(
                component=component,
                state=HealthState.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.monotonic() - start) * 1000,
                last_check=time.monotonic(),
            )
            logger.warning(f"Health check failed for {component}: {e}")
