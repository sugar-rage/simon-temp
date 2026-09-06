"""Navigation pipeline — orchestrates GPS → Routing → Guidance.

This is the top-level navigation component. It owns the GPS provider,
router, geocoder, and guidance engine, providing a unified API for
the rest of SIMON.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.models.location import GeoLocation, Route
from core.config.system_config import NavigationConfig
from core.capabilities.registry import CapabilityRegistry
from navigation.gps.providers import BaseGPSProvider, GPSFilter
from navigation.routing.routers import BaseRouter, Geocoder
from navigation.guidance.guidance_engine import GuidanceEngine

logger = logging.getLogger("simon.navigation.pipeline")


class NavigationPipeline:
    """Top-level navigation orchestrator.

    Parameters
    ----------
    gps : BaseGPSProvider
        GPS provider (injected).
    router : BaseRouter
        Routing engine (injected).
    event_bus : EventBus
        For publishing navigation events.
    config : NavigationConfig, optional
        Navigation configuration.
    capabilities : CapabilityRegistry, optional
        For registering capabilities.
    """

    def __init__(
        self,
        gps: BaseGPSProvider,
        router: BaseRouter,
        event_bus: EventBus,
        config: Optional[NavigationConfig] = None,
        capabilities: Optional[CapabilityRegistry] = None,
    ) -> None:
        self._config = config or NavigationConfig()
        self._event_bus = event_bus
        self._capabilities = capabilities

        # Components
        self._gps = gps
        self._gps_filter = GPSFilter()
        self._router = router
        self._geocoder = Geocoder()
        self._guidance = GuidanceEngine(
            event_bus=event_bus,
            config=self._config.guidance,
        )

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_location: Optional[GeoLocation] = None
        self._current_route: Optional[Route] = None

    @property
    def latest_location(self) -> Optional[GeoLocation]:
        return self._latest_location

    @property
    def current_route(self) -> Optional[Route]:
        return self._current_route

    @property
    def is_navigating(self) -> bool:
        return self._guidance.is_active

    def start(self) -> bool:
        """Start the GPS and navigation processing loop."""
        if self._running:
            return True

        # Start GPS
        gps_ok = self._gps.start()
        if self._capabilities:
            self._capabilities.register(
                "capability.gps",
                available=gps_ok,
                reason=None if gps_ok else "GPS unavailable",
            )
            self._capabilities.register(
                "capability.navigation",
                available=self._router.is_available(),
                reason=None if self._router.is_available() else "Router unavailable",
            )

        if not gps_ok:
            logger.error("GPS failed to start")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop, name="NavigationPipeline", daemon=True
        )
        self._thread.start()
        logger.info("Navigation pipeline started")
        return True

    def stop(self) -> None:
        """Stop the navigation pipeline."""
        self._running = False
        self._guidance.stop_navigation()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._gps.stop()
        logger.info("Navigation pipeline stopped")

    def navigate_to(self, destination: GeoLocation) -> Optional[Route]:
        """Compute a route and start navigation.

        Parameters
        ----------
        destination : GeoLocation
            Target location.

        Returns
        -------
        Route or None
            The computed route, or None on failure.
        """
        if not self._latest_location:
            logger.error("Cannot navigate: no GPS fix")
            return None

        route = self._router.compute_route(
            self._latest_location, destination
        )
        if route is None:
            logger.error("Route computation failed")
            return None

        self._current_route = route
        self._guidance.start_navigation(route)

        self._event_bus.publish(Event(
            topic=event_types.NAV_ROUTE_COMPUTED,
            data={
                "total_distance_m": route.total_distance_m,
                "total_duration_s": route.total_duration_s,
                "steps": len(route.steps),
            },
            priority=Priority.NAVIGATION,
            source="navigation_pipeline",
        ))

        return route

    def navigate_to_address(self, address: str) -> Optional[Route]:
        """Geocode an address and navigate to it."""
        location = self._geocoder.geocode(address)
        if location is None:
            logger.error("Geocoding failed for: %s", address)
            return None
        return self.navigate_to(location)

    def cancel_navigation(self) -> None:
        """Cancel the current navigation."""
        self._guidance.stop_navigation()
        self._current_route = None
        logger.info("Navigation cancelled")

    def _update_loop(self) -> None:
        """GPS polling loop — runs on dedicated thread."""
        while self._running:
            raw = self._gps.get_location()
            if raw is not None:
                filtered = self._gps_filter.update(raw)
                self._latest_location = filtered

                # Publish position event
                self._event_bus.publish(Event(
                    topic=event_types.NAV_POSITION,
                    data={
                        "latitude": filtered.latitude,
                        "longitude": filtered.longitude,
                    },
                    priority=Priority.INFORMATIONAL,
                    source="navigation_pipeline",
                ))

                # Update guidance
                if self._guidance.is_active:
                    announcement = self._guidance.update(filtered)
                    if announcement:
                        self._event_bus.publish(Event(
                            topic=event_types.LOGIC_ANNOUNCE,
                            data={
                                "text": announcement,
                                "priority": Priority.NAVIGATION,
                            },
                            priority=Priority.NAVIGATION,
                            source="navigation_pipeline",
                        ))

            time.sleep(1.0)  # GPS polling interval
