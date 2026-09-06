"""Capability Registry — single source of truth for runtime feature availability.

Replaces scattered ``try/import/except`` blocks and ``is_available`` booleans
with a centralized, thread-safe registry.

See implementation_plan.md Section 3.11 for the full design.

Known capability names::

    capability.camera            — Camera device accessible
    capability.detection         — YOLO model loaded
    capability.ocr               — OCR engine available
    capability.face_recognition  — InsightFace loaded
    capability.scene_analysis    — Scene analyzer ready
    capability.depth_estimation  — MiDaS model loaded (optional)
    capability.gps               — GPS device responding
    capability.routing_online    — OSRM server reachable
    capability.routing_offline   — osmnx + NetworkX available
    capability.speech            — SpeechManager running
    capability.speaker_verify    — Speaker verification model loaded
    capability.plugins           — Plugin system active

Usage::

    registry = CapabilityRegistry()
    registry.register("capability.camera", available=True)
    registry.register("capability.ocr", available=False, reason="not installed")

    if registry.is_available("capability.ocr"):
        result = ocr_engine.read_text(frame)
    else:
        speech.speak("Text reading is not available")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("simon.core.capabilities")

# Type alias for change listener
CapabilityChangeListener = Callable[[str, "CapabilityStatus"], None]


@dataclass
class CapabilityStatus:
    """Status of a single capability.

    Parameters
    ----------
    available : bool
        Whether the capability is currently usable.
    reason : str, optional
        Human-readable reason for unavailability (e.g. "insightface not installed").
    degraded : bool
        True if available but with reduced functionality.
    last_checked : float
        Timestamp of the last status update.
    metadata : dict
        Additional metadata (e.g. model version, device name).
    """

    available: bool = False
    reason: Optional[str] = None
    degraded: bool = False
    last_checked: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class CapabilityRegistry:
    """Thread-safe registry for runtime capability discovery.

    Subsystems register their capabilities at init time and update them
    on status changes (e.g. camera disconnected, model evicted).

    Other subsystems query the registry instead of doing their own
    ``try/import`` checks.

    Parameters
    ----------
    None — construct directly or use dependency injection.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityStatus] = {}
        self._lock = threading.Lock()
        self._listeners: list[CapabilityChangeListener] = []

    def register(
        self,
        name: str,
        *,
        available: bool = False,
        reason: Optional[str] = None,
        degraded: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """Register or update a capability.

        Parameters
        ----------
        name : str
            Capability identifier (e.g. ``"capability.camera"``).
        available : bool
            Whether the capability is currently usable.
        reason : str, optional
            Reason for unavailability.
        degraded : bool
            True if available but limited.
        metadata : dict, optional
            Extra information (model version, device info, etc.).
        """
        status = CapabilityStatus(
            available=available,
            reason=reason,
            degraded=degraded,
            last_checked=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            old = self._capabilities.get(name)
            self._capabilities[name] = status
            listeners = list(self._listeners)

        # Log and notify
        if old is None:
            logger.info(
                "Capability registered: %s = %s%s",
                name,
                "available" if available else "unavailable",
                f" (degraded)" if degraded else "",
            )
        elif old.available != available or old.degraded != degraded:
            logger.info(
                "Capability changed: %s = %s%s%s",
                name,
                "available" if available else "unavailable",
                f" (degraded)" if degraded else "",
                f" ({reason})" if reason else "",
            )

        for listener in listeners:
            try:
                listener(name, status)
            except Exception:
                logger.exception(
                    "Listener error for capability change: %s", name
                )

    def update(
        self,
        name: str,
        *,
        available: Optional[bool] = None,
        reason: Optional[str] = None,
        degraded: Optional[bool] = None,
    ) -> None:
        """Update an existing capability's status.

        Only provided fields are updated; others are preserved.
        If the capability doesn't exist, this acts like ``register()``.
        """
        with self._lock:
            existing = self._capabilities.get(name)

        if existing is None:
            self.register(
                name,
                available=available if available is not None else False,
                reason=reason,
                degraded=degraded if degraded is not None else False,
            )
            return

        self.register(
            name,
            available=(
                available if available is not None else existing.available
            ),
            reason=reason if reason is not None else existing.reason,
            degraded=(
                degraded if degraded is not None else existing.degraded
            ),
            metadata=existing.metadata,
        )

    def is_available(self, name: str) -> bool:
        """Check if a capability is available (not degraded, not unavailable)."""
        with self._lock:
            status = self._capabilities.get(name)
            return status is not None and status.available

    def is_degraded(self, name: str) -> bool:
        """Check if a capability is in degraded mode."""
        with self._lock:
            status = self._capabilities.get(name)
            return status is not None and status.degraded

    def get_status(self, name: str) -> Optional[CapabilityStatus]:
        """Return the full status of a capability, or None if not registered."""
        with self._lock:
            status = self._capabilities.get(name)
            if status is None:
                return None
            # Return a copy to prevent mutation
            return CapabilityStatus(
                available=status.available,
                reason=status.reason,
                degraded=status.degraded,
                last_checked=status.last_checked,
                metadata=dict(status.metadata),
            )

    def list_available(self) -> list[str]:
        """Return names of all currently available capabilities."""
        with self._lock:
            return [
                name
                for name, status in self._capabilities.items()
                if status.available and not status.degraded
            ]

    def list_degraded(self) -> list[str]:
        """Return names of capabilities in degraded mode."""
        with self._lock:
            return [
                name
                for name, status in self._capabilities.items()
                if status.degraded
            ]

    def list_unavailable(self) -> list[str]:
        """Return names of capabilities that are unavailable."""
        with self._lock:
            return [
                name
                for name, status in self._capabilities.items()
                if not status.available
            ]

    def list_all(self) -> dict[str, CapabilityStatus]:
        """Return a snapshot of all registered capabilities."""
        with self._lock:
            return dict(self._capabilities)

    def subscribe_changes(self, callback: CapabilityChangeListener) -> None:
        """Register a listener for capability changes.

        The callback receives ``(name, new_status)`` and is called outside
        the lock to prevent deadlocks.
        """
        with self._lock:
            self._listeners.append(callback)

    def unsubscribe_changes(self, callback: CapabilityChangeListener) -> None:
        """Remove a capability change listener."""
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def summary(self) -> str:
        """Return a human-readable summary for diagnostics.

        Example output::

            Capabilities:
              ✅ capability.camera (available)
              ⚠️ capability.ocr (degraded: easyocr fallback)
              ❌ capability.gps (unavailable: device not found)
        """
        with self._lock:
            items = sorted(self._capabilities.items())

        if not items:
            return "Capabilities: none registered"

        lines = ["Capabilities:"]
        for name, status in items:
            if status.available and not status.degraded:
                icon = "✅"
                label = "available"
            elif status.degraded:
                icon = "⚠️"
                label = f"degraded: {status.reason}" if status.reason else "degraded"
            else:
                icon = "❌"
                label = (
                    f"unavailable: {status.reason}"
                    if status.reason
                    else "unavailable"
                )
            lines.append(f"  {icon} {name} ({label})")

        return "\n".join(lines)
