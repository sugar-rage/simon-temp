"""Location, waypoint, and route data models for navigation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.models.enums import NavigationManeuver


@dataclass
class GeoLocation:
    """A GPS position with accuracy and motion metadata."""

    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy_m: float = 0.0
    heading: Optional[float] = None
    speed_mps: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        """Alias for ``accuracy_m`` (used by GPSFilter)."""
        return self.accuracy_m

    def distance_to(self, other: GeoLocation) -> float:
        """Approximate distance in meters using Haversine formula."""
        import math

        R = 6_371_000  # Earth radius in meters
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlat = math.radians(other.latitude - self.latitude)
        dlon = math.radians(other.longitude - self.longitude)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


@dataclass
class Waypoint:
    """A named geographic point on a route."""

    location: GeoLocation
    name: Optional[str] = None
    index: int = 0


@dataclass
class RouteStep:
    """A single turn-by-turn instruction within a route."""

    instruction: str
    """Human-readable instruction, e.g. 'Turn left onto Main Street'."""

    distance_m: float
    """Distance in meters to this maneuver point."""

    duration_s: float = 0.0
    """Estimated duration in seconds for this step."""

    maneuver: str = NavigationManeuver.STRAIGHT.value
    """Maneuver type (``"turn-left"``, ``"straight"``, ``"arrive"``, etc.)."""

    street_name: Optional[str] = None
    waypoint: Optional[GeoLocation] = None

    location: Optional[GeoLocation] = None
    """Geographic location of this maneuver point (for distance checks)."""

    @property
    def maneuver_enum(self) -> NavigationManeuver:
        """Return the maneuver as an enum, defaulting to STRAIGHT."""
        try:
            return NavigationManeuver(self.maneuver)
        except ValueError:
            return NavigationManeuver.STRAIGHT


@dataclass
class Route:
    """A computed navigation route from origin to destination."""

    origin: Optional[GeoLocation] = None
    destination: Optional[GeoLocation] = None
    waypoints: list[GeoLocation] = field(default_factory=list)
    steps: list[RouteStep] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_duration_s: float = 0.0
    destination_name: str = ""
    computed_at: float = field(default_factory=time.time)

    # Backward-compatible aliases
    @property
    def distance_m(self) -> float:
        return self.total_distance_m

    @property
    def duration_s(self) -> float:
        return self.total_duration_s

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        return len(self.waypoints) == 0

