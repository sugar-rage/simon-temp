"""System Context — maintains shared runtime context across subsystems.

Provides a single, thread-safe context object that aggregates:
- Current WorldModel from Vision
- Current GeoLocation from Navigation
- Current SystemState
- Active capabilities
- User preferences and session data
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.models.location import GeoLocation
from core.state.state_machine import SystemState
from vision.pipeline.perception_fusion import WorldModel
from vision.scene.scene_analyzer import SceneAnalysis

logger = logging.getLogger("simon.core.context")


@dataclass
class SpatialContext:
    """Current spatial awareness context.

    Attributes
    ----------
    scene_type : str
        Current scene classification.
    is_indoors : bool
        Whether the user appears to be indoors.
    is_moving : bool
        Whether the user appears to be moving.
    dominant_objects : list[str]
        Most common objects in the current scene.
    """

    scene_type: str = "unknown"
    is_indoors: bool = False
    is_moving: bool = False
    dominant_objects: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class NavigationContext:
    """Current navigation context.

    Attributes
    ----------
    active : bool
        Whether navigation is currently active.
    destination : str
        Current navigation destination name.
    distance_remaining_m : float
        Distance remaining to destination (meters).
    current_instruction : str
        Current navigation instruction.
    """

    active: bool = False
    destination: str = ""
    distance_remaining_m: float = 0.0
    current_instruction: str = ""
    on_route: bool = True


class SystemContext:
    """Thread-safe shared context across all SIMON subsystems.

    This is the central "knowledge" of the system — what SIMON currently
    knows about the world, where it is, and what it's doing.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._world: Optional[WorldModel] = None
        self._location: Optional[GeoLocation] = None
        self._spatial = SpatialContext()
        self._navigation = NavigationContext()
        self._system_state = SystemState.STARTING
        self._user_preferences: dict[str, Any] = {}
        self._session_start = time.time()

    # ── WorldModel ───────────────────────────────────────────────────

    @property
    def world(self) -> Optional[WorldModel]:
        with self._lock:
            return self._world

    def update_world(self, world: WorldModel) -> None:
        """Update the current WorldModel."""
        with self._lock:
            self._world = world
            # Update spatial context from scene analysis
            if world.scene:
                self._spatial.scene_type = world.scene.scene_type
                self._spatial.is_indoors = world.scene.scene_type == "indoor"
                if world.scene.object_counts:
                    self._spatial.dominant_objects = sorted(
                        world.scene.object_counts.keys(),
                        key=lambda k: world.scene.object_counts[k],
                        reverse=True,
                    )[:5]
                self._spatial.last_updated = time.time()

    # ── Location ─────────────────────────────────────────────────────

    @property
    def location(self) -> Optional[GeoLocation]:
        with self._lock:
            return self._location

    def update_location(self, location: GeoLocation) -> None:
        with self._lock:
            self._location = location

    # ── Spatial ──────────────────────────────────────────────────────

    @property
    def spatial(self) -> SpatialContext:
        with self._lock:
            return self._spatial

    # ── Navigation ───────────────────────────────────────────────────

    @property
    def navigation(self) -> NavigationContext:
        with self._lock:
            return self._navigation

    def update_navigation(
        self,
        active: Optional[bool] = None,
        destination: Optional[str] = None,
        distance_remaining_m: Optional[float] = None,
        current_instruction: Optional[str] = None,
        on_route: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if active is not None:
                self._navigation.active = active
            if destination is not None:
                self._navigation.destination = destination
            if distance_remaining_m is not None:
                self._navigation.distance_remaining_m = distance_remaining_m
            if current_instruction is not None:
                self._navigation.current_instruction = current_instruction
            if on_route is not None:
                self._navigation.on_route = on_route

    # ── System State ─────────────────────────────────────────────────

    @property
    def system_state(self) -> SystemState:
        with self._lock:
            return self._system_state

    def update_system_state(self, state: SystemState) -> None:
        with self._lock:
            self._system_state = state

    # ── User Preferences ─────────────────────────────────────────────

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._user_preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        with self._lock:
            self._user_preferences[key] = value

    # ── Session ──────────────────────────────────────────────────────

    @property
    def session_duration_s(self) -> float:
        return time.time() - self._session_start

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current system context."""
        with self._lock:
            return {
                "system_state": self._system_state.name,
                "session_duration_s": self.session_duration_s,
                "has_world": self._world is not None,
                "entity_count": self._world.entity_count if self._world else 0,
                "scene_type": self._spatial.scene_type,
                "has_location": self._location is not None,
                "navigation_active": self._navigation.active,
            }
