"""Unit tests for Watchdog — heartbeat monitoring and stale detection."""

from __future__ import annotations

import time
import pytest

from core.events.event_bus import EventBus
from core.health.watchdog import Watchdog


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    bus.start()
    yield bus
    bus.stop()
    bus.clear()


@pytest.fixture
def watchdog(event_bus):
    wd = Watchdog(event_bus=event_bus, interval_s=0.1, timeout_s=0.3)
    yield wd
    if wd.is_running:
        wd.stop()


class TestWatchdog:
    def test_register_thread(self, watchdog):
        watchdog.register_thread("test_thread")
        threads = watchdog.get_registered_threads()
        assert "test_thread" in threads

    def test_heartbeat_updates_timestamp(self, watchdog):
        watchdog.register_thread("test_thread")
        t1 = watchdog.get_registered_threads()["test_thread"]
        time.sleep(0.05)
        watchdog.heartbeat("test_thread")
        t2 = watchdog.get_registered_threads()["test_thread"]
        assert t2 > t1

    def test_unregister_thread(self, watchdog):
        watchdog.register_thread("temp")
        watchdog.unregister_thread("temp")
        assert "temp" not in watchdog.get_registered_threads()

    def test_start_stop_lifecycle(self, watchdog):
        watchdog.start()
        assert watchdog.is_running
        watchdog.stop()
        assert not watchdog.is_running

    def test_stale_heartbeat_fires_event(self, watchdog, event_bus):
        """Thread with stale heartbeat should trigger health.thread_dead."""
        dead_events = []
        event_bus.subscribe(
            "health.thread_dead",
            lambda e: dead_events.append(e),
            source="test",
        )

        watchdog.register_thread("stale_thread")
        watchdog.start()

        # Don't send any heartbeats — wait for timeout + check interval
        time.sleep(0.6)  # timeout=0.3 + interval=0.1 + margin
        watchdog.stop()

        assert len(dead_events) >= 1
        assert dead_events[0].data["thread_name"] == "stale_thread"

    def test_active_heartbeat_no_event(self, watchdog, event_bus):
        """Thread with active heartbeat should NOT trigger health.thread_dead."""
        dead_events = []
        event_bus.subscribe(
            "health.thread_dead",
            lambda e: dead_events.append(e),
            source="test",
        )

        watchdog.register_thread("active_thread")
        watchdog.start()

        # Send heartbeats faster than the timeout
        for _ in range(5):
            watchdog.heartbeat("active_thread")
            time.sleep(0.05)

        watchdog.stop()
        assert len(dead_events) == 0

    def test_multiple_threads_independent(self, watchdog, event_bus):
        """Stale detection should be per-thread."""
        dead_threads = []
        event_bus.subscribe(
            "health.thread_dead",
            lambda e: dead_threads.append(e.data["thread_name"]),
            source="test",
        )

        watchdog.register_thread("healthy")
        watchdog.register_thread("stale")
        watchdog.start()

        # Only heartbeat "healthy"
        for _ in range(5):
            watchdog.heartbeat("healthy")
            time.sleep(0.1)

        watchdog.stop()
        assert "stale" in dead_threads
        assert "healthy" not in dead_threads

    def test_double_start_is_safe(self, watchdog):
        watchdog.start()
        watchdog.start()  # Should not raise or create second thread
        assert watchdog.is_running
        watchdog.stop()
