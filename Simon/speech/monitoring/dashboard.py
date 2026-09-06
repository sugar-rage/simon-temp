"""
Dashboard — TUI status dashboard for the SIMON speech subsystem.

Provides a simple text-based overview of system status:
- Component health states
- Pipeline activity (listen/speak running?)
- Recent metrics (STT latency, confidence, queue depth)
- Active scene classification
- Context state

This is designed for terminal/logging output, not a full GUI.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from speech.monitoring.health_monitor import HealthMonitor, HealthState

logger = logging.getLogger(__name__)


class Dashboard:
    """Text-based system status dashboard.

    Collects status from multiple sources and formats a human-readable
    report.  Can be called periodically or on-demand.

    Args:
        health_monitor: HealthMonitor for component status.
    """

    def __init__(self, health_monitor: Optional[HealthMonitor] = None):
        self._health = health_monitor or HealthMonitor()
        self._custom_panels: Dict[str, Dict[str, Any]] = {}
        self._pipeline_status: Dict[str, str] = {
            "listen": "stopped",
            "speak": "stopped",
        }

    def set_pipeline_status(self, pipeline: str, status: str) -> None:
        """Update pipeline status (e.g. 'listen' → 'running')."""
        self._pipeline_status[pipeline] = status

    def add_panel(self, name: str, data: Dict[str, Any]) -> None:
        """Add a custom data panel to the dashboard.

        Args:
            name: Panel title.
            data: Key-value pairs to display.
        """
        self._custom_panels[name] = data

    def render(self) -> str:
        """Render the dashboard as a text string.

        Returns:
            Multi-line formatted status report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  SIMON Speech Subsystem Dashboard")
        lines.append("=" * 60)
        lines.append("")

        # Pipeline status
        lines.append("─── Pipelines ───")
        for name, status in self._pipeline_status.items():
            icon = "🟢" if status == "running" else "🔴"
            lines.append(f"  {icon} {name.capitalize()}: {status}")
        lines.append("")

        # Health status
        lines.append("─── Component Health ───")
        health_results = self._health.check()
        for name, status in sorted(health_results.items()):
            icon = {
                HealthState.HEALTHY: "✅",
                HealthState.DEGRADED: "⚠️",
                HealthState.UNHEALTHY: "❌",
                HealthState.UNKNOWN: "❓",
            }.get(status.state, "❓")
            msg = f" — {status.message}" if status.message else ""
            lines.append(f"  {icon} {name}: {status.state.name}{msg}")
        lines.append("")

        # Custom panels
        for panel_name, data in self._custom_panels.items():
            lines.append(f"─── {panel_name} ───")
            for key, value in data.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def log_status(self) -> None:
        """Log the dashboard output at INFO level."""
        logger.info(f"\n{self.render()}")

    def get_health_summary(self) -> Dict[str, str]:
        """Get component health as a simple dict."""
        return self._health.get_summary()
