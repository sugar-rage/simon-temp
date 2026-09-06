"""Abstract GPS provider and concrete implementations.

Includes:
- BaseGPSProvider: ABC for all GPS sources
- SerialGPS: Real GPS via serial NMEA
- SimulationGPS: Simulated GPS for development/testing
- GPSFilter: Kalman-like smoothing filter for noisy GPS readings
"""

from __future__ import annotations

import abc
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.models.location import GeoLocation

logger = logging.getLogger("simon.navigation.gps")


# ── Abstract Base ────────────────────────────────────────────────────


class BaseGPSProvider(abc.ABC):
    """Abstract GPS provider."""

    @abc.abstractmethod
    def start(self) -> bool:
        """Start receiving GPS updates. Return True on success."""
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop receiving GPS updates."""
        ...

    @abc.abstractmethod
    def get_location(self) -> Optional[GeoLocation]:
        """Return the latest GPS location, or None if unavailable."""
        ...

    @abc.abstractmethod
    def has_fix(self) -> bool:
        """Return True if we have a valid GPS fix."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Serial GPS (Real Hardware) ───────────────────────────────────────


class SerialGPS(BaseGPSProvider):
    """GPS via serial NMEA (e.g. USB GPS dongle).

    Parses NMEA $GPGGA sentences from a serial port.

    Parameters
    ----------
    port : str
        Serial port (e.g. ``/dev/ttyUSB0``, ``COM3``).
    baud_rate : int
        Serial baud rate.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud_rate: int = 9600) -> None:
        self._port = port
        self._baud_rate = baud_rate
        self._serial: Any = None
        self._location: Optional[GeoLocation] = None
        self._has_fix = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baud_rate, timeout=1)
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, name="GPS-Serial", daemon=True
            )
            self._thread.start()
            logger.info("Serial GPS started on %s @ %d", self._port, self._baud_rate)
            return True
        except ImportError:
            logger.error("pyserial not installed")
            return False
        except Exception as e:
            logger.error("Failed to open serial GPS: %s", e)
            return False

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._serial:
            self._serial.close()
        logger.info("Serial GPS stopped")

    def get_location(self) -> Optional[GeoLocation]:
        with self._lock:
            return self._location

    def has_fix(self) -> bool:
        with self._lock:
            return self._has_fix

    def _read_loop(self) -> None:
        while self._running and self._serial:
            try:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    self._parse_gga(line)
            except Exception as e:
                logger.debug("GPS read error: %s", e)
                time.sleep(0.1)

    def _parse_gga(self, sentence: str) -> None:
        """Parse a GPGGA/GNGGA NMEA sentence."""
        try:
            parts = sentence.split(",")
            if len(parts) < 10:
                return

            fix_quality = int(parts[6]) if parts[6] else 0
            if fix_quality == 0:
                with self._lock:
                    self._has_fix = False
                return

            lat = self._nmea_to_decimal(parts[2], parts[3])
            lon = self._nmea_to_decimal(parts[4], parts[5])
            alt = float(parts[9]) if parts[9] else None
            accuracy = float(parts[8]) if parts[8] else None  # HDOP

            with self._lock:
                self._location = GeoLocation(
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    accuracy=accuracy,
                )
                self._has_fix = True

        except (ValueError, IndexError) as e:
            logger.debug("GGA parse error: %s", e)

    @staticmethod
    def _nmea_to_decimal(value: str, direction: str) -> float:
        """Convert NMEA coordinate to decimal degrees."""
        if not value:
            return 0.0
        # NMEA: DDDMM.MMMMM
        point_pos = value.index(".")
        degrees = float(value[: point_pos - 2])
        minutes = float(value[point_pos - 2 :])
        result = degrees + minutes / 60.0
        if direction in ("S", "W"):
            result = -result
        return result


# ── Simulation GPS ───────────────────────────────────────────────────


class SimulationGPS(BaseGPSProvider):
    """Simulated GPS that replays a predefined route.

    Parameters
    ----------
    waypoints : list[GeoLocation]
        Route waypoints to replay.
    speed_m_s : float
        Simulated movement speed (meters/second).
    loop : bool
        Whether to loop the route.
    """

    def __init__(
        self,
        waypoints: Optional[list[GeoLocation]] = None,
        speed_m_s: float = 1.5,
        loop: bool = False,
    ) -> None:
        self._waypoints = waypoints or []
        self._speed = speed_m_s
        self._loop = loop
        self._current_index = 0
        self._location: Optional[GeoLocation] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not self._waypoints:
            logger.warning("SimulationGPS: no waypoints provided")
            return False

        self._running = True
        self._current_index = 0
        self._location = self._waypoints[0]
        self._thread = threading.Thread(
            target=self._simulate_loop, name="GPS-Sim", daemon=True
        )
        self._thread.start()
        logger.info("Simulation GPS started (%d waypoints)", len(self._waypoints))
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Simulation GPS stopped")

    def get_location(self) -> Optional[GeoLocation]:
        return self._location

    def has_fix(self) -> bool:
        return self._location is not None and self._running

    def set_location(self, location: GeoLocation) -> None:
        """Manually set the simulated location (for testing)."""
        self._location = location

    def _simulate_loop(self) -> None:
        while self._running:
            if self._current_index < len(self._waypoints) - 1:
                self._current_index += 1
                self._location = self._waypoints[self._current_index]
            elif self._loop:
                self._current_index = 0
                self._location = self._waypoints[0]
            else:
                self._running = False
                break

            # Compute delay based on distance between waypoints
            if self._current_index > 0:
                prev = self._waypoints[self._current_index - 1]
                curr = self._waypoints[self._current_index]
                dist = prev.distance_to(curr)
                delay = max(dist / self._speed, 0.1) if self._speed > 0 else 1.0
            else:
                delay = 1.0

            time.sleep(delay)


# ── GPS Filter ───────────────────────────────────────────────────────


class GPSFilter:
    """Simple GPS smoothing filter.

    Uses exponential moving average to smooth noisy GPS readings.

    Parameters
    ----------
    alpha : float
        Smoothing factor (0.0-1.0). Lower = more smoothing.
    min_accuracy_m : float
        Discard readings with accuracy worse than this (meters).
    """

    def __init__(self, alpha: float = 0.3, min_accuracy_m: float = 50.0) -> None:
        self._alpha = alpha
        self._min_accuracy = min_accuracy_m
        self._filtered: Optional[GeoLocation] = None
        self._reading_count = 0

    def update(self, raw: GeoLocation) -> GeoLocation:
        """Apply smoothing to a raw GPS reading.

        Returns the filtered location.
        """
        self._reading_count += 1

        # Reject very inaccurate readings
        if raw.accuracy is not None and raw.accuracy > self._min_accuracy:
            if self._filtered is not None:
                return self._filtered
            # No filtered value yet, use raw anyway
            self._filtered = raw
            return raw

        if self._filtered is None:
            self._filtered = raw
            return raw

        # Exponential moving average
        new_lat = self._alpha * raw.latitude + (1 - self._alpha) * self._filtered.latitude
        new_lon = self._alpha * raw.longitude + (1 - self._alpha) * self._filtered.longitude
        new_alt = raw.altitude  # Don't smooth altitude

        self._filtered = GeoLocation(
            latitude=new_lat,
            longitude=new_lon,
            altitude=new_alt,
            accuracy_m=raw.accuracy,
        )
        return self._filtered

    @property
    def filtered_location(self) -> Optional[GeoLocation]:
        return self._filtered

    @property
    def reading_count(self) -> int:
        return self._reading_count

    def reset(self) -> None:
        self._filtered = None
        self._reading_count = 0
