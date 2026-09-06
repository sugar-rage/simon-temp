"""Unit tests for Phase 4 — Navigation Subsystem.

Tests: GPS providers, GPSFilter, OfflineRouter, GuidanceEngine,
OffRouteDetector, InstructionGenerator, NavigationPipeline.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.location import GeoLocation, Route, RouteStep
from core.models.enums import Priority
from core.metrics.collector import SystemMetricsCollector
from navigation.gps.providers import (
    BaseGPSProvider, SimulationGPS, GPSFilter,
)
from navigation.routing.routers import BaseRouter, OfflineRouter
from navigation.guidance.guidance_engine import (
    GuidanceEngine, OffRouteDetector, InstructionGenerator,
)


# ── Helpers ──────────────────────────────────────────────────────────


def loc(lat: float, lon: float) -> GeoLocation:
    """Create a test GeoLocation."""
    return GeoLocation(latitude=lat, longitude=lon)


def make_route(
    origin: GeoLocation | None = None,
    destination: GeoLocation | None = None,
    n_steps: int = 3,
) -> Route:
    """Create a test route."""
    o = origin or loc(40.0, -74.0)
    d = destination or loc(40.01, -74.0)

    # Generate intermediate waypoints
    waypoints = [o]
    for i in range(1, n_steps):
        frac = i / n_steps
        waypoints.append(loc(
            o.latitude + frac * (d.latitude - o.latitude),
            o.longitude + frac * (d.longitude - o.longitude),
        ))
    waypoints.append(d)

    steps = []
    for i, wp in enumerate(waypoints[:-1]):
        steps.append(RouteStep(
            instruction=f"Continue step {i}",
            distance_m=wp.distance_to(waypoints[i + 1]),
            duration_s=10.0,
            maneuver="continue",
            location=wp,
        ))

    return Route(
        origin=o,
        destination=d,
        waypoints=waypoints,
        steps=steps,
        total_distance_m=o.distance_to(d),
        total_duration_s=n_steps * 10.0,
    )


class MockRouter(BaseRouter):
    """Mock router that returns a preset route."""

    def __init__(self, route: Route | None = None, available: bool = True) -> None:
        self._route = route
        self._available = available

    def compute_route(self, origin, destination, profile="foot"):
        if self._route:
            return self._route
        return make_route(origin, destination)

    def is_available(self) -> bool:
        return self._available


@pytest.fixture(autouse=True)
def reset_metrics():
    yield
    SystemMetricsCollector.reset_instance()


@pytest.fixture
def event_bus():
    bus = EventBus(enable_metrics=False)
    yield bus
    bus.clear()


# ═════════════════════════════════════════════════════════════════════
# SimulationGPS Tests
# ═════════════════════════════════════════════════════════════════════


class TestSimulationGPS:
    def test_start_with_waypoints(self):
        gps = SimulationGPS(waypoints=[loc(40.0, -74.0), loc(40.001, -74.0)])
        assert gps.start()
        assert gps.has_fix()
        gps.stop()

    def test_start_without_waypoints(self):
        gps = SimulationGPS(waypoints=[])
        assert not gps.start()

    def test_get_location(self):
        gps = SimulationGPS(waypoints=[loc(40.0, -74.0)])
        gps.start()
        location = gps.get_location()
        assert location is not None
        assert location.latitude == pytest.approx(40.0)
        gps.stop()

    def test_set_location(self):
        gps = SimulationGPS(waypoints=[loc(0, 0)])
        gps.set_location(loc(50.0, 10.0))
        assert gps.get_location().latitude == pytest.approx(50.0)

    def test_no_fix_after_stop(self):
        gps = SimulationGPS(waypoints=[loc(40.0, -74.0)])
        gps.start()
        gps.stop()
        assert not gps.has_fix()


# ═════════════════════════════════════════════════════════════════════
# GPSFilter Tests
# ═════════════════════════════════════════════════════════════════════


class TestGPSFilter:
    def test_first_reading_passes_through(self):
        f = GPSFilter()
        result = f.update(loc(40.0, -74.0))
        assert result.latitude == pytest.approx(40.0)
        assert f.reading_count == 1

    def test_smoothing_applied(self):
        f = GPSFilter(alpha=0.5)
        f.update(loc(40.0, -74.0))
        result = f.update(loc(40.1, -74.0))
        # Should be between 40.0 and 40.1
        assert 40.0 < result.latitude < 40.1

    def test_reject_inaccurate_reading(self):
        f = GPSFilter(min_accuracy_m=10.0)
        f.update(loc(40.0, -74.0))
        # Feed an inaccurate reading
        bad = GeoLocation(latitude=50.0, longitude=-74.0, accuracy_m=100.0)
        result = f.update(bad)
        # Should return previous filtered value, not the bad reading
        assert result.latitude == pytest.approx(40.0)

    def test_reset(self):
        f = GPSFilter()
        f.update(loc(40.0, -74.0))
        f.reset()
        assert f.filtered_location is None
        assert f.reading_count == 0


# ═════════════════════════════════════════════════════════════════════
# OfflineRouter Tests
# ═════════════════════════════════════════════════════════════════════


class TestOfflineRouter:
    def test_no_graph_returns_straight_line(self):
        router = OfflineRouter()
        route = router.compute_route(loc(40.0, -74.0), loc(40.01, -74.0))
        assert route is not None
        assert route.total_distance_m > 0
        assert len(route.steps) >= 1

    def test_not_available_without_graph(self):
        router = OfflineRouter()
        assert not router.is_available()

    def test_graph_from_json(self, tmp_path):
        import json
        graph = {
            "nodes": [
                {"id": "a", "lat": 40.0, "lon": -74.0},
                {"id": "b", "lat": 40.005, "lon": -74.0},
                {"id": "c", "lat": 40.01, "lon": -74.0},
            ],
            "edges": [
                {"from": "a", "to": "b", "weight": 500},
                {"from": "b", "to": "c", "weight": 500},
            ],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(graph))

        router = OfflineRouter(graph_file=str(path))
        assert router.is_available()

        route = router.compute_route(loc(40.0, -74.0), loc(40.01, -74.0))
        assert route is not None
        assert len(route.steps) >= 1


# ═════════════════════════════════════════════════════════════════════
# OffRouteDetector Tests
# ═════════════════════════════════════════════════════════════════════


class TestOffRouteDetector:
    def test_on_route(self):
        detector = OffRouteDetector(threshold_m=50.0)
        route = make_route()
        is_off, dist = detector.check(route.waypoints[0], route)
        assert not is_off

    def test_off_route_detected(self):
        detector = OffRouteDetector(threshold_m=50.0, consecutive_threshold=1)
        route = make_route()
        # Location very far from route
        far = loc(45.0, -80.0)
        is_off, dist = detector.check(far, route)
        assert is_off
        assert dist > 50

    def test_consecutive_threshold(self):
        detector = OffRouteDetector(threshold_m=50.0, consecutive_threshold=3)
        route = make_route()
        far = loc(45.0, -80.0)
        # First two: not yet off route
        assert not detector.check(far, route)[0]
        assert not detector.check(far, route)[0]
        # Third: now off route
        assert detector.check(far, route)[0]

    def test_reset_clears_state(self):
        detector = OffRouteDetector(threshold_m=50.0, consecutive_threshold=1)
        route = make_route()
        far = loc(45.0, -80.0)
        detector.check(far, route)
        detector.reset()
        assert not detector.is_off_route


# ═════════════════════════════════════════════════════════════════════
# InstructionGenerator Tests
# ═════════════════════════════════════════════════════════════════════


class TestInstructionGenerator:
    def test_turn_left(self):
        gen = InstructionGenerator()
        step = RouteStep(
            instruction="Main St",
            distance_m=100,
            maneuver="turn-left",
            location=loc(40.0, -74.0),
        )
        text = gen.format_step(step, 30.0)
        assert "Turn left" in text or "turn left" in text

    def test_close_distance_says_now(self):
        gen = InstructionGenerator()
        step = RouteStep(
            instruction="Main St",
            distance_m=10,
            maneuver="turn-right",
            location=loc(40.0, -74.0),
        )
        text = gen.format_step(step, 5.0)
        assert "now" in text.lower()

    def test_arrival(self):
        gen = InstructionGenerator()
        text = gen.format_arrival("Central Park")
        assert "Central Park" in text

    def test_off_route(self):
        gen = InstructionGenerator()
        text = gen.format_off_route(30.0)
        assert "30" in text

    def test_reroute(self):
        gen = InstructionGenerator()
        text = gen.format_reroute()
        assert "recalculated" in text.lower()


# ═════════════════════════════════════════════════════════════════════
# GuidanceEngine Tests
# ═════════════════════════════════════════════════════════════════════


class TestGuidanceEngine:
    def test_not_active_initially(self, event_bus):
        engine = GuidanceEngine(event_bus=event_bus)
        assert not engine.is_active

    def test_start_navigation(self, event_bus):
        engine = GuidanceEngine(event_bus=event_bus)
        route = make_route()
        engine.start_navigation(route)
        assert engine.is_active
        assert engine.current_route is route

    def test_stop_navigation(self, event_bus):
        engine = GuidanceEngine(event_bus=event_bus)
        engine.start_navigation(make_route())
        engine.stop_navigation()
        assert not engine.is_active

    def test_arrival_detection(self, event_bus):
        from core.config.system_config import GuidanceConfig
        config = GuidanceConfig(arrival_threshold_m=100.0)
        engine = GuidanceEngine(event_bus=event_bus, config=config)

        route = make_route()
        engine.start_navigation(route)

        # Move to destination
        result = engine.update(route.destination)
        assert result is not None
        assert "arrived" in result.lower()
        assert not engine.is_active

    def test_off_route_announcement(self, event_bus):
        from core.config.system_config import GuidanceConfig
        config = GuidanceConfig(off_route_threshold_m=50.0)
        engine = GuidanceEngine(event_bus=event_bus, config=config)
        engine._off_route._consecutive_threshold = 1  # immediate

        route = make_route()
        engine.start_navigation(route)

        result = engine.update(loc(45.0, -80.0))  # very far
        assert result is not None
        assert "off route" in result.lower()

    def test_remaining_steps(self, event_bus):
        engine = GuidanceEngine(event_bus=event_bus)
        route = make_route(n_steps=5)
        engine.start_navigation(route)
        assert engine.remaining_steps == 5

    def test_instruction_near_step(self, event_bus):
        from core.config.system_config import GuidanceConfig
        config = GuidanceConfig(announce_distance_m=1000.0)
        engine = GuidanceEngine(event_bus=event_bus, config=config)

        route = make_route()
        engine.start_navigation(route)

        # Stand near the first step
        result = engine.update(route.steps[0].location)
        # Should get an announcement
        assert result is not None

    def test_events_published(self, event_bus):
        from core.config.system_config import GuidanceConfig
        config = GuidanceConfig(arrival_threshold_m=100.0)

        received = []
        event_bus.subscribe("navigation.*", lambda e: received.append(e))

        engine = GuidanceEngine(event_bus=event_bus, config=config)
        route = make_route()
        engine.start_navigation(route)
        engine.update(route.destination)  # arrive
        assert len(received) > 0


# ═════════════════════════════════════════════════════════════════════
# NavigationPipeline Tests
# ═════════════════════════════════════════════════════════════════════


class TestNavigationPipeline:
    def test_start_and_stop(self, event_bus):
        from navigation.pipeline import NavigationPipeline

        gps = SimulationGPS(waypoints=[loc(40.0, -74.0), loc(40.001, -74.0)])
        router = MockRouter()
        pipeline = NavigationPipeline(gps=gps, router=router, event_bus=event_bus)
        assert pipeline.start()
        time.sleep(0.5)
        assert pipeline.latest_location is not None
        pipeline.stop()

    def test_navigate_to(self, event_bus):
        from navigation.pipeline import NavigationPipeline

        gps = SimulationGPS(waypoints=[loc(40.0, -74.0)])
        gps.start()
        router = MockRouter()
        pipeline = NavigationPipeline(gps=gps, router=router, event_bus=event_bus)
        pipeline._latest_location = loc(40.0, -74.0)  # simulate fix

        route = pipeline.navigate_to(loc(40.01, -74.0))
        assert route is not None
        assert pipeline.is_navigating
        pipeline.stop()

    def test_navigate_without_fix_fails(self, event_bus):
        from navigation.pipeline import NavigationPipeline

        gps = SimulationGPS(waypoints=[loc(40.0, -74.0)])
        router = MockRouter()
        pipeline = NavigationPipeline(gps=gps, router=router, event_bus=event_bus)
        # No fix yet
        result = pipeline.navigate_to(loc(40.01, -74.0))
        assert result is None

    def test_cancel_navigation(self, event_bus):
        from navigation.pipeline import NavigationPipeline

        gps = SimulationGPS(waypoints=[loc(40.0, -74.0)])
        router = MockRouter()
        pipeline = NavigationPipeline(gps=gps, router=router, event_bus=event_bus)
        pipeline._latest_location = loc(40.0, -74.0)
        pipeline.navigate_to(loc(40.01, -74.0))
        pipeline.cancel_navigation()
        assert not pipeline.is_navigating
        pipeline.stop()
