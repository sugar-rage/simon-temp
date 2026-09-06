"""
SIMON Navigation Logic — OSM-based outdoor routing with voice guidance.

Uses OpenStreetMap data via osmnx for route computation and
geopy for geocoding place names to coordinates.

Features:
  - Geocode any place name to coordinates (via Nominatim)
  - Download local road network around current location
  - Compute shortest path using NetworkX
  - Generate step-by-step street-by-street instructions
  - Voice feedback through the shared VoiceEngine

Usage:
    from logic.navigation_logic import NavigationLogic
    nav = NavigationLogic(voice_engine=tts)
    nav.navigate_to("Central Library, Bangalore", current_coords=(12.97, 77.59))
"""

import threading

try:
    import config
except ImportError:
    config = None

try:
    import osmnx as ox
    import networkx as nx
    from geopy.geocoders import Nominatim
    _NAV_AVAILABLE = True
except ImportError as e:
    _NAV_AVAILABLE = False
    _NAV_IMPORT_ERROR = str(e)


class NavigationLogic:
    """OSM-based navigation with voice-guided turn-by-turn instructions."""

    def __init__(self, voice_engine=None):
        """
        Initialize the navigation module.

        Args:
            voice_engine: VoiceEngine instance for spoken directions.
        """
        self.voice = voice_engine
        self._graph = None
        self._graph_center = None
        self._graph_radius = 0
        self._route = None
        self._instructions = []
        self._navigating = False
        self._nav_thread = None
        self._nav_lock = threading.Lock()

        # Geocoder
        if _NAV_AVAILABLE:
            _user_agent = getattr(config, "NOMINATIM_USER_AGENT", "SIMON-Navigation/1.0") if config else "SIMON-Navigation/1.0"
            self._geolocator = Nominatim(user_agent=_user_agent, timeout=10)
        else:
            self._geolocator = None

    @property
    def is_available(self):
        """Check if navigation dependencies are installed."""
        return _NAV_AVAILABLE

    @property
    def is_navigating(self):
        return self._navigating

    # ------------------------------------------------------------------
    # Geocoding
    # ------------------------------------------------------------------
    def get_coords(self, location_name):
        """
        Convert a place name / address to (latitude, longitude).

        Args:
            location_name: e.g. "Central Library, Bangalore"

        Returns:
            tuple: (lat, lon) or None if not found.
        """
        if not self._geolocator:
            return None

        try:
            location = self._geolocator.geocode(location_name, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"[Navigation] Geocoding error: {e}")

        return None

    # ------------------------------------------------------------------
    # Graph & Routing
    # ------------------------------------------------------------------
    def build_graph(self, current_coords, radius=2000):
        """
        Download the local road network around the given coordinates.

        Args:
            current_coords: (lat, lon) tuple.
            radius: Search radius in meters (default 2km).
        """
        try:
            print(f"[Navigation] Downloading road network ({radius}m radius)...")
            ox.settings.timeout = 30
            self._graph = ox.graph_from_point(
                current_coords, dist=radius, network_type="walk"
            )
            self._graph_center = current_coords
            self._graph_radius = radius
            print(f"[Navigation] Road network loaded OK")
        except Exception as e:
            print(f"[Navigation] Failed to load road network: {e}")
            self._graph = None
            self._graph_center = None
            self._graph_radius = 0

    def shortest_route(self, start_coords, end_coords):
        """
        Compute the shortest path between two coordinates.

        Args:
            start_coords: (lat, lon) of origin.
            end_coords:   (lat, lon) of destination.

        Returns:
            list: Route node IDs, or None on failure.
        """
        try:
            from geopy.distance import geodesic
            dist_m = geodesic(start_coords, end_coords).meters
        except Exception as e:
            print(f"[Navigation] Distance calculation error: {e}")
            dist_m = 2000

        # Add 20% margin to ensure the route fits inside the downloaded bounding box
        required_radius = max(2000, int(dist_m * 1.2))

        if required_radius > 10000:  # 10km max to prevent massive OSM downloads
            print(f"[Navigation] Destination too far ({int(dist_m)}m). Max supported radius is 10km.")
            self._say("The destination is too far. I can only navigate to places within 8 kilometers.")
            return None

        # Check if the requested route is completely covered by the cached graph
        needs_rebuild = True
        if self._graph is not None and self._graph_center is not None:
            try:
                from geopy.distance import geodesic
                dist_to_center = geodesic(self._graph_center, start_coords).meters
                
                # If distance to the cached graph's center + new required radius 
                # is less than the cached graph's radius, the whole requested area is inside it.
                if dist_to_center + required_radius <= self._graph_radius:
                    needs_rebuild = False
                    print(f"[Navigation] Cache hit! Reusing existing {self._graph_radius}m graph.")
            except Exception as e:
                pass

        if needs_rebuild:
            self.build_graph(start_coords, radius=required_radius)

        if self._graph is None:
            return None

        try:
            orig_node = ox.distance.nearest_nodes(
                self._graph, start_coords[1], start_coords[0]
            )
            dest_node = ox.distance.nearest_nodes(
                self._graph, end_coords[1], end_coords[0]
            )
            route = nx.shortest_path(
                self._graph, orig_node, dest_node, weight="length"
            )
            return route
        except Exception as e:
            print(f"[Navigation] Routing error: {e}")
            return None

    def route_instructions(self, route):
        """
        Extract human-readable step-by-step directions from a route.

        Args:
            route: List of node IDs from shortest_route().

        Returns:
            list: Strings like "Step 1: Go along Main Street"
        """
        if self._graph is None or route is None:
            return []

        try:
            # Iterate consecutive node pairs to extract street names
            # (compatible with all osmnx versions, handles MultiDiGraph)
            instructions = []
            prev_street = None
            step_num = 1

            for u, v in zip(route[:-1], route[1:]):
                edge_data = self._graph.get_edge_data(u, v)
                if edge_data is None:
                    continue
                # MultiDiGraph: edge_data is {key: {attrs}}; pick first available key
                if isinstance(edge_data, dict):
                    first_key = next(iter(edge_data), None)
                    if first_key is not None and isinstance(edge_data[first_key], dict):
                        attrs = edge_data[first_key]
                    else:
                        attrs = edge_data
                else:
                    attrs = edge_data if isinstance(edge_data, dict) else {}

                street = attrs.get("name", None) if isinstance(attrs, dict) else None
                # name can be a list for merged edges
                if isinstance(street, list):
                    street = street[0] if street else None

                if street != prev_street:
                    if street:
                        instructions.append(f"Step {step_num}: Go along {street}")
                    else:
                        instructions.append(f"Step {step_num}: Continue on unnamed road")
                    prev_street = street
                    step_num += 1

            return instructions
        except Exception as e:
            print(f"[Navigation] Instruction generation error: {e}")
            return []

    # ------------------------------------------------------------------
    # High-Level Navigation
    # ------------------------------------------------------------------
    def navigate_to(self, destination_name, current_coords=None):
        """
        Start navigation to a destination (runs in background thread).

        Args:
            destination_name: Place name, e.g. "Central Library, Bangalore"
            current_coords:   (lat, lon) of current position. If None,
                              uses a default (Bangalore center).
        """
        with self._nav_lock:
            if not _NAV_AVAILABLE:
                self._say(f"Navigation unavailable. Install osmnx, networkx, geopy.")
                print(f"[Navigation] Missing dependency: {_NAV_IMPORT_ERROR}")
                return

            if self._navigating:
                self._say("Already navigating. Say cancel to stop.")
                return

            # Default coords (Bangalore center) if no GPS
            if current_coords is None:
                current_coords = (12.9716, 77.5946)
                self._say("I don't have your current location. Navigation may be inaccurate.")
                print("[Navigation] No GPS — using default location (Bangalore)")

            # Run in background to avoid blocking the main loop
            self._navigating = True
            self._nav_thread = threading.Thread(
                target=self._navigate_worker,
                args=(destination_name, current_coords),
                daemon=True,
            )
            self._nav_thread.start()

    def cancel(self):
        """Cancel ongoing navigation."""
        with self._nav_lock:
            if self._navigating:
                self._navigating = False
                self._say("Navigation cancelled.")
                print("[Navigation] Cancelled.")

    def _navigate_worker(self, destination_name, current_coords):
        """Background worker for navigation computation + voice guidance."""
        try:
            self._say(f"Finding route to {destination_name}")

            # Geocode destination
            dest_coords = self.get_coords(destination_name)
            if not self._navigating:
                return
            if not dest_coords:
                self._say(f"Could not find {destination_name}")
                self._navigating = False
                return

            print(f"[Navigation] Destination: {dest_coords}")

            # Build graph and compute route
            route = self.shortest_route(current_coords, dest_coords)
            if not self._navigating:
                return
            if not route:
                self._say("Could not find a route.")
                self._navigating = False
                return

            # Generate instructions
            instructions = self.route_instructions(route)
            if not self._navigating:
                return
            self._route = route
            self._instructions = instructions

            if not instructions:
                self._say("Route found but no street names available.")
                self._navigating = False
                return

            # Speak each instruction
            self._say(f"Route found. {len(instructions)} steps.")
            for step in instructions:
                if not self._navigating:  # Cancelled
                    break
                self._say(step)

            if self._navigating:
                self._say("You have arrived at your destination.")

        except Exception as e:
            print(f"[Navigation] Error: {e}")
            self._say("Navigation error occurred.")
        finally:
            with self._nav_lock:
                self._navigating = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _say(self, text):
        """Speak text if voice engine is available."""
        print(f"[Navigation] {text}")
        if self.voice:
            self.voice.speak(text, priority=2)
