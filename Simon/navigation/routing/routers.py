"""Abstract routing engine and concrete implementations.

Includes:
- BaseRouter: ABC for route computation
- OSRMRouter: Online OSRM routing API
- OfflineRouter: Offline graph-based routing using OSM data
- Geocoder: Destination name → GeoLocation resolution
"""

from __future__ import annotations

import abc
import json
import logging
import os
from typing import Any, Optional

from core.models.location import GeoLocation, Route, RouteStep

logger = logging.getLogger("simon.navigation.routing")


# ── Abstract Base ────────────────────────────────────────────────────


class BaseRouter(abc.ABC):
    """Abstract routing engine."""

    @abc.abstractmethod
    def compute_route(
        self,
        origin: GeoLocation,
        destination: GeoLocation,
        profile: str = "foot",
    ) -> Optional[Route]:
        """Compute a route between two points.

        Parameters
        ----------
        origin : GeoLocation
            Starting location.
        destination : GeoLocation
            Destination location.
        profile : str
            Routing profile (``"foot"``, ``"car"``, ``"bike"``).

        Returns
        -------
        Route or None
            Computed route, or None on failure.
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the routing backend is reachable/ready."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── OSRM Router (Online) ────────────────────────────────────────────


class OSRMRouter(BaseRouter):
    """Online routing via the OSRM HTTP API.

    Parameters
    ----------
    base_url : str
        OSRM server URL.
    timeout_s : float
        HTTP request timeout.
    """

    def __init__(
        self,
        base_url: str = "http://router.project-osrm.org",
        timeout_s: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    def compute_route(
        self,
        origin: GeoLocation,
        destination: GeoLocation,
        profile: str = "foot",
    ) -> Optional[Route]:
        try:
            import urllib.request

            url = (
                f"{self._base_url}/route/v1/{profile}/"
                f"{origin.longitude},{origin.latitude};"
                f"{destination.longitude},{destination.latitude}"
                f"?overview=full&steps=true&geometries=geojson"
            )

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())

            if data.get("code") != "Ok":
                logger.warning("OSRM error: %s", data.get("message", "unknown"))
                return None

            route_data = data["routes"][0]
            legs = route_data.get("legs", [])

            steps: list[RouteStep] = []
            for leg in legs:
                for step_data in leg.get("steps", []):
                    maneuver = step_data.get("maneuver", {})
                    location = maneuver.get("location", [0, 0])
                    steps.append(RouteStep(
                        instruction=step_data.get("name", "Continue"),
                        distance_m=step_data.get("distance", 0),
                        duration_s=step_data.get("duration", 0),
                        maneuver=maneuver.get("type", "continue"),
                        location=GeoLocation(
                            latitude=location[1],
                            longitude=location[0],
                        ),
                    ))

            # Extract geometry
            geometry_coords = route_data.get("geometry", {}).get("coordinates", [])
            waypoints = [
                GeoLocation(latitude=c[1], longitude=c[0])
                for c in geometry_coords
            ]

            return Route(
                origin=origin,
                destination=destination,
                steps=steps,
                total_distance_m=route_data.get("distance", 0),
                total_duration_s=route_data.get("duration", 0),
                waypoints=waypoints,
            )

        except Exception as e:
            logger.error("OSRM routing failed: %s", e)
            return None

    def is_available(self) -> bool:
        try:
            import urllib.request
            url = f"{self._base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0):
                return True
        except Exception:
            # OSRM public doesn't have /health, try a minimal route instead
            try:
                import urllib.request
                url = f"{self._base_url}/route/v1/foot/0,0;0,0"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3.0):
                    return True
            except Exception:
                return False


# ── Offline Router (Graph-based) ────────────────────────────────────


class OfflineRouter(BaseRouter):
    """Offline routing using local OSM data.

    Uses a simplified A* algorithm on a pre-built graph.
    This is a fallback router for when OSRM is unavailable.

    Parameters
    ----------
    graph_file : str
        Path to a JSON graph file (nodes + edges).
    """

    def __init__(self, graph_file: Optional[str] = None) -> None:
        self._graph_file = graph_file
        self._nodes: dict[str, GeoLocation] = {}
        self._edges: dict[str, list[tuple[str, float]]] = {}
        self._loaded = False

        if graph_file and os.path.exists(graph_file):
            self._load_graph(graph_file)

    def compute_route(
        self,
        origin: GeoLocation,
        destination: GeoLocation,
        profile: str = "foot",
    ) -> Optional[Route]:
        if not self._loaded:
            logger.warning("Offline router: no graph loaded")
            return self._straight_line_route(origin, destination)

        # Find nearest nodes
        start_id = self._nearest_node(origin)
        end_id = self._nearest_node(destination)

        if not start_id or not end_id:
            return self._straight_line_route(origin, destination)

        # A* search
        path = self._astar(start_id, end_id)
        if not path:
            return self._straight_line_route(origin, destination)

        # Build route from path
        waypoints = [self._nodes[nid] for nid in path]
        total_dist = sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        )

        steps = []
        for i, wp in enumerate(waypoints[:-1]):
            steps.append(RouteStep(
                instruction="Continue",
                distance_m=wp.distance_to(waypoints[i + 1]),
                duration_s=wp.distance_to(waypoints[i + 1]) / 1.4,  # ~walking speed
                maneuver="continue",
                location=wp,
            ))

        return Route(
            origin=origin,
            destination=destination,
            steps=steps,
            total_distance_m=total_dist,
            total_duration_s=total_dist / 1.4,
            waypoints=waypoints,
        )

    def is_available(self) -> bool:
        return self._loaded

    def _straight_line_route(
        self, origin: GeoLocation, destination: GeoLocation
    ) -> Route:
        """Fallback: straight-line route with no turn-by-turn."""
        dist = origin.distance_to(destination)
        return Route(
            origin=origin,
            destination=destination,
            steps=[
                RouteStep(
                    instruction="Head toward destination",
                    distance_m=dist,
                    duration_s=dist / 1.4,
                    maneuver="continue",
                    location=origin,
                )
            ],
            total_distance_m=dist,
            total_duration_s=dist / 1.4,
            waypoints=[origin, destination],
        )

    def _load_graph(self, filepath: str) -> None:
        try:
            with open(filepath) as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                nid = str(node["id"])
                self._nodes[nid] = GeoLocation(
                    latitude=node["lat"], longitude=node["lon"]
                )
            for edge in data.get("edges", []):
                src = str(edge["from"])
                dst = str(edge["to"])
                weight = edge.get("weight", 1.0)
                self._edges.setdefault(src, []).append((dst, weight))
                if not edge.get("oneway", False):
                    self._edges.setdefault(dst, []).append((src, weight))
            self._loaded = True
            logger.info("Loaded offline graph: %d nodes, %d edges",
                        len(self._nodes), sum(len(v) for v in self._edges.values()))
        except Exception as e:
            logger.error("Failed to load offline graph: %s", e)

    def _nearest_node(self, location: GeoLocation) -> Optional[str]:
        if not self._nodes:
            return None
        return min(
            self._nodes.keys(),
            key=lambda nid: location.distance_to(self._nodes[nid]),
        )

    def _astar(self, start: str, end: str) -> Optional[list[str]]:
        """A* shortest path search."""
        import heapq

        if start == end:
            return [start]

        open_set: list[tuple[float, str]] = [(0.0, start)]
        came_from: dict[str, str] = {}
        g_score: dict[str, float] = {start: 0.0}

        end_loc = self._nodes.get(end)
        if not end_loc:
            return None

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor, weight in self._edges.get(current, []):
                tentative = g_score.get(current, float("inf")) + weight
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    # Heuristic: straight-line distance to end
                    n_loc = self._nodes.get(neighbor)
                    h = n_loc.distance_to(end_loc) if n_loc else 0.0
                    heapq.heappush(open_set, (tentative + h, neighbor))

        return None  # No path found


# ── Geocoder ─────────────────────────────────────────────────────────


class Geocoder:
    """Converts place names to GeoLocation via Nominatim API.

    Parameters
    ----------
    user_agent : str
        User-Agent for Nominatim API (required by terms of use).
    """

    def __init__(self, user_agent: str = "SIMON/1.0") -> None:
        self._user_agent = user_agent

    def geocode(self, query: str) -> Optional[GeoLocation]:
        """Convert a place name to coordinates.

        Returns None if the place is not found or the API is unreachable.
        """
        try:
            import urllib.request
            import urllib.parse

            url = (
                f"https://nominatim.openstreetmap.org/search?"
                f"q={urllib.parse.quote(query)}&format=json&limit=1"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": self._user_agent}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode())

            if not data:
                return None

            result = data[0]
            return GeoLocation(
                latitude=float(result["lat"]),
                longitude=float(result["lon"]),
            )

        except Exception as e:
            logger.error("Geocoding failed for '%s': %s", query, e)
            return None

    def reverse_geocode(self, location: GeoLocation) -> Optional[str]:
        """Convert coordinates to a place name."""
        try:
            import urllib.request

            url = (
                f"https://nominatim.openstreetmap.org/reverse?"
                f"lat={location.latitude}&lon={location.longitude}"
                f"&format=json"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": self._user_agent}
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode())

            return data.get("display_name")

        except Exception as e:
            logger.error("Reverse geocoding failed: %s", e)
            return None
