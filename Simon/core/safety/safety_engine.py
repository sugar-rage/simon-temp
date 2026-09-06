"""Safety Engine — central safety coordinator.

Evaluates the WorldModel for safety hazards and generates appropriate
responses (TTS warnings, event bus alerts). This is the synchronous
safety layer that runs on every frame with detections.

Safety pipeline::

    WorldModel → HazardClassifier → SafetyRules → EmergencyHandler → Events

Design principles:
- Safety checks NEVER raise exceptions (fail-open with logging).
- Synchronous dispatch for zero-latency hazard warnings.
- Configurable safety rules for different contexts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority, HazardLevel
from core.config.system_config import SafetyConfig
from vision.pipeline.perception_fusion import Entity, WorldModel

logger = logging.getLogger("simon.core.safety")


@dataclass
class HazardAlert:
    """A classified hazard ready for response.

    Attributes
    ----------
    entity : Entity
        The entity that triggered the hazard.
    level : int
        Hazard level (HazardLevel enum value).
    message : str
        Human-readable hazard description.
    timestamp : float
        When the hazard was classified.
    """

    entity: Entity
    level: int
    message: str
    timestamp: float = field(default_factory=time.time)


class HazardClassifier:
    """Classifies entities into hazard levels based on distance and class.

    Parameters
    ----------
    vehicle_classes : set[str]
        Object classes considered vehicles.
    critical_distance : str
        Distance value that triggers CRITICAL.
    """

    def __init__(
        self,
        vehicle_classes: Optional[set[str]] = None,
        critical_distance: str = "near",
    ) -> None:
        self._vehicle_classes = vehicle_classes or {
            "car", "truck", "bus", "motorcycle", "bicycle",
        }
        self._critical_distance = critical_distance

    def classify(self, entity: Entity) -> Optional[int]:
        """Return HazardLevel for an entity, or None if not hazardous."""
        if entity.cls not in self._vehicle_classes:
            return None

        if entity.distance == self._critical_distance:
            return HazardLevel.CRITICAL
        elif entity.distance == "medium":
            return HazardLevel.WARNING
        elif entity.distance == "far":
            return HazardLevel.CAUTION

        return HazardLevel.CAUTION


class SafetyRules:
    """Configurable safety rules engine.

    Determines whether a hazard should trigger an alert based on
    cooldowns, suppression rules, and context.

    Parameters
    ----------
    alert_cooldown_s : float
        Minimum seconds between alerts for the same hazard class.
    """

    def __init__(self, alert_cooldown_s: float = 3.0) -> None:
        self._cooldown_s = alert_cooldown_s
        self._last_alert: dict[str, float] = {}

    def should_alert(self, entity: Entity, hazard_level: int) -> bool:
        """Return True if this hazard should trigger an alert."""
        # CRITICAL hazards always alert
        if hazard_level <= HazardLevel.CRITICAL:
            return True

        # Check cooldown for non-critical hazards
        key = f"{entity.cls}:{entity.position or 'unknown'}"
        now = time.time()
        last = self._last_alert.get(key, 0.0)

        if now - last < self._cooldown_s:
            return False

        self._last_alert[key] = now
        return True

    def reset(self) -> None:
        """Clear all cooldown tracking."""
        self._last_alert.clear()


class EmergencyHandler:
    """Handles emergency-level hazards.

    Generates immediate TTS and event bus alerts for CRITICAL hazards.
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._event_bus = event_bus
        self._emergency_count = 0

    def handle(self, alert: HazardAlert) -> None:
        """Process an emergency hazard alert."""
        self._emergency_count += 1

        if self._event_bus:
            self._event_bus.publish(Event(
                topic=event_types.SAFETY_EMERGENCY,
                data={
                    "message": alert.message,
                    "cls": alert.entity.cls,
                    "position": alert.entity.position,
                    "distance": alert.entity.distance,
                    "hazard_level": alert.level,
                },
                priority=Priority.EMERGENCY,
                source="safety_engine",
            ))

        logger.critical("EMERGENCY: %s", alert.message)

    @property
    def emergency_count(self) -> int:
        return self._emergency_count


class SafetyEngine:
    """Central safety coordinator.

    Orchestrates HazardClassifier → SafetyRules → EmergencyHandler
    on every WorldModel update.

    Parameters
    ----------
    config : SafetyConfig
        Safety configuration.
    event_bus : EventBus, optional
        Event bus for alerts.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._config = config or SafetyConfig()
        self._enabled = self._config.enabled

        vehicle_classes = set(self._config.vehicle_classes)
        self._classifier = HazardClassifier(vehicle_classes=vehicle_classes)
        self._rules = SafetyRules(alert_cooldown_s=self._config.hazard_cooldown_s)
        self._emergency = EmergencyHandler(event_bus=event_bus)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def emergency_count(self) -> int:
        return self._emergency.emergency_count

    def evaluate(self, world: WorldModel) -> list[HazardAlert]:
        """Evaluate the WorldModel for safety hazards.

        Returns a list of hazard alerts that passed safety rules.
        """
        if not self._enabled:
            return []

        alerts: list[HazardAlert] = []

        try:
            for entity in world.entities:
                hazard_level = self._classifier.classify(entity)
                if hazard_level is None:
                    continue

                if not self._rules.should_alert(entity, hazard_level):
                    continue

                # Build description
                position = entity.position or "unknown"
                distance = entity.distance or ""
                message = (
                    f"{'Warning' if hazard_level <= HazardLevel.CRITICAL else 'Caution'}: "
                    f"{entity.cls} {position}"
                )
                if distance:
                    message += f", {distance}"

                alert = HazardAlert(
                    entity=entity,
                    level=hazard_level,
                    message=message,
                )
                alerts.append(alert)

                # Emergency-level hazards get immediate handling
                if hazard_level <= HazardLevel.CRITICAL:
                    self._emergency.handle(alert)

        except Exception as e:
            # Safety code NEVER raises — fail open
            logger.error("Safety evaluation failed: %s", e)

        return alerts
