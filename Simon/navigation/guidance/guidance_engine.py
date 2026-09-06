"""Guidance Engine — turn-by-turn navigation with voice announcements.

Includes:
- GuidanceEngine: Tracks position along a route and generates instructions
- InstructionGenerator: Converts RouteSteps into speech-ready text
- OffRouteDetector: Detects when the user deviates from the route
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.models.location import GeoLocation, Route, RouteStep
from core.config.system_config import GuidanceConfig, NavigationConfig

logger = logging.getLogger("simon.navigation.guidance")


# ── Off-Route Detector ───────────────────────────────────────────────


class OffRouteDetector:
    """Detects when the user deviates from the planned route.

    Parameters
    ----------
    threshold_m : float
        Distance from route (meters) to trigger off-route alert.
    consecutive_threshold : int
        Number of consecutive off-route readings before alerting.
    """

    def __init__(
        self,
        threshold_m: float = 25.0,
        consecutive_threshold: int = 3,
    ) -> None:
        self._threshold = threshold_m
        self._consecutive_threshold = consecutive_threshold
        self._off_route_count = 0
        self._is_off_route = False

    def check(self, location: GeoLocation, route: Route) -> tuple[bool, float]:
        """Check if the user is off route.

        Returns
        -------
        tuple[bool, float]
            (is_off_route, distance_from_route_m).
        """
        # Find minimum distance to any route waypoint
        min_dist = float("inf")
        for wp in route.waypoints:
            dist = location.distance_to(wp)
            if dist < min_dist:
                min_dist = dist

        if min_dist > self._threshold:
            self._off_route_count += 1
        else:
            self._off_route_count = 0
            self._is_off_route = False

        if self._off_route_count >= self._consecutive_threshold:
            self._is_off_route = True

        return self._is_off_route, min_dist

    @property
    def is_off_route(self) -> bool:
        return self._is_off_route

    def reset(self) -> None:
        self._off_route_count = 0
        self._is_off_route = False


# ── Instruction Generator ────────────────────────────────────────────


class InstructionGenerator:
    """Converts RouteSteps into speech-friendly navigation text."""

    _MANEUVER_MAP = {
        "turn-left": "Turn left",
        "turn-right": "Turn right",
        "turn-sharp-left": "Turn sharp left",
        "turn-sharp-right": "Turn sharp right",
        "turn-slight-left": "Bear left",
        "turn-slight-right": "Bear right",
        "continue": "Continue straight",
        "roundabout": "Enter the roundabout",
        "depart": "Head out",
        "arrive": "You have arrived",
        "uturn": "Make a U-turn",
        "merge": "Merge",
        "fork-left": "Keep left at the fork",
        "fork-right": "Keep right at the fork",
    }

    def format_step(self, step: RouteStep, distance_to_step_m: float) -> str:
        """Format a route step for TTS announcement.

        Parameters
        ----------
        step : RouteStep
            The upcoming route step.
        distance_to_step_m : float
            Distance remaining to the step's location.
        """
        action = self._MANEUVER_MAP.get(step.maneuver, step.maneuver)

        if distance_to_step_m < 15:
            return f"{action} now"
        elif distance_to_step_m < 50:
            return f"In {int(distance_to_step_m)} meters, {action.lower()}"
        elif distance_to_step_m < 200:
            return f"In {int(distance_to_step_m)} meters, {action.lower()}"
        else:
            return f"In {int(distance_to_step_m)} meters, {action.lower()}"

    def format_arrival(self, destination_name: str = "your destination") -> str:
        return f"You have arrived at {destination_name}"

    def format_off_route(self, distance_m: float) -> str:
        return f"You appear to be off route by {int(distance_m)} meters. Recalculating."

    def format_reroute(self) -> str:
        return "Route recalculated. Follow the new directions."


# ── Guidance Engine ──────────────────────────────────────────────────


class GuidanceEngine:
    """Turn-by-turn navigation guidance engine.

    Tracks the user's position along a route, generates upcoming
    instructions, detects off-route conditions, and publishes events.

    Parameters
    ----------
    event_bus : EventBus
        For publishing navigation events.
    config : GuidanceConfig
        Guidance-specific configuration.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[GuidanceConfig] = None,
    ) -> None:
        self._event_bus = event_bus
        self._config = config or GuidanceConfig()
        self._route: Optional[Route] = None
        self._current_step_index: int = 0
        self._off_route = OffRouteDetector(
            threshold_m=self._config.off_route_threshold_m,
        )
        self._instructions = InstructionGenerator()
        self._last_announcement_step: int = -1
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active and self._route is not None

    @property
    def current_route(self) -> Optional[Route]:
        return self._route

    @property
    def current_step(self) -> Optional[RouteStep]:
        if self._route and self._current_step_index < len(self._route.steps):
            return self._route.steps[self._current_step_index]
        return None

    @property
    def current_step_index(self) -> int:
        return self._current_step_index

    @property
    def remaining_steps(self) -> int:
        if not self._route:
            return 0
        return max(0, len(self._route.steps) - self._current_step_index)

    def start_navigation(self, route: Route) -> None:
        """Start navigating a route."""
        self._route = route
        self._current_step_index = 0
        self._active = True
        self._off_route.reset()
        self._last_announcement_step = -1
        logger.info(
            "Navigation started: %d steps, %.0f m",
            len(route.steps), route.total_distance_m,
        )

    def stop_navigation(self) -> None:
        """Stop navigation."""
        self._active = False
        self._route = None
        self._off_route.reset()
        logger.info("Navigation stopped")

    def update(self, location: GeoLocation) -> Optional[str]:
        """Process a GPS update and return an announcement (if any).

        Parameters
        ----------
        location : GeoLocation
            Current user position.

        Returns
        -------
        str or None
            Speech text to announce, or None.
        """
        if not self.is_active or not self._route:
            return None

        # Check arrival
        dist_to_dest = location.distance_to(self._route.destination)
        if dist_to_dest < self._config.arrival_threshold_m:
            self._active = False
            text = self._instructions.format_arrival()
            self._publish_event(event_types.NAV_ARRIVED, {
                "destination": str(self._route.destination),
            })
            return text

        # Check off-route
        is_off, off_dist = self._off_route.check(location, self._route)
        if is_off:
            text = self._instructions.format_off_route(off_dist)
            self._publish_event(event_types.NAV_OFF_ROUTE, {
                "distance_m": off_dist,
            })
            return text

        # Advance step pointer
        self._advance_step(location)

        # Generate instruction if approaching next step
        step = self.current_step
        if step and step.location:
            dist = location.distance_to(step.location)
            if (
                dist < self._config.announce_distance_m
                and self._current_step_index != self._last_announcement_step
            ):
                self._last_announcement_step = self._current_step_index
                text = self._instructions.format_step(step, dist)
                self._publish_event(event_types.NAV_INSTRUCTION, {
                    "instruction": text,
                    "distance_m": dist,
                    "maneuver": step.maneuver,
                    "step_index": self._current_step_index,
                })
                return text

        return None

    def _advance_step(self, location: GeoLocation) -> None:
        """Advance the step pointer if the user has passed the current step."""
        if not self._route:
            return

        while self._current_step_index < len(self._route.steps) - 1:
            step = self._route.steps[self._current_step_index]
            if step.location:
                dist = location.distance_to(step.location)
                if dist < 10:  # passed the step
                    self._current_step_index += 1
                else:
                    break
            else:
                self._current_step_index += 1

    def _publish_event(self, topic: str, data: dict) -> None:
        if self._event_bus:
            self._event_bus.publish_sync(Event(
                topic=topic,
                data=data,
                priority=Priority.NAVIGATION,
                source="guidance_engine",
            ))
