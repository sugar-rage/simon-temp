"""Unit tests for the Event Bus — pub/sub, wildcards, priorities, safety dispatch."""

from __future__ import annotations

import threading
import time

import pytest

from core.events.event_bus import EventBus
from core.events.filters import PriorityFilter, TopicFilter, RateLimiter
from core.models.events import Event
from core.models.enums import Priority


# ── Subscription and Dispatch ────────────────────────────────────────


class TestEventBusSubscription:
    def test_subscribe_and_publish_sync(self, event_bus):
        received = []
        event_bus.subscribe("test.event", lambda e: received.append(e))
        event = Event(topic="test.event", data={"x": 1})
        event_bus.publish_sync(event)
        assert len(received) == 1
        assert received[0].data["x"] == 1

    def test_no_match_not_delivered(self, event_bus):
        received = []
        event_bus.subscribe("other.topic", lambda e: received.append(e))
        event_bus.publish_sync(Event(topic="test.event"))
        assert len(received) == 0

    def test_wildcard_subscription(self, event_bus):
        received = []
        event_bus.subscribe("vision.*", lambda e: received.append(e))
        event_bus.publish_sync(Event(topic="vision.detection"))
        event_bus.publish_sync(Event(topic="vision.face"))
        event_bus.publish_sync(Event(topic="navigation.gps"))
        assert len(received) == 2

    def test_global_wildcard(self, event_bus):
        received = []
        event_bus.subscribe("*", lambda e: received.append(e))
        event_bus.publish_sync(Event(topic="anything.at.all"))
        assert len(received) == 1

    def test_multiple_subscribers(self, event_bus):
        results = {"a": [], "b": []}
        event_bus.subscribe("test.event", lambda e: results["a"].append(e))
        event_bus.subscribe("test.event", lambda e: results["b"].append(e))
        event_bus.publish_sync(Event(topic="test.event"))
        assert len(results["a"]) == 1
        assert len(results["b"]) == 1

    def test_unsubscribe(self, event_bus):
        received = []
        handler = lambda e: received.append(e)
        event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe(handler)
        event_bus.publish_sync(Event(topic="test.event"))
        assert len(received) == 0

    def test_subscriber_count(self, event_bus):
        assert event_bus.subscriber_count == 0
        event_bus.subscribe("a", lambda e: None)
        event_bus.subscribe("b", lambda e: None)
        assert event_bus.subscriber_count == 2


# ── Priority Filtering ───────────────────────────────────────────────


class TestEventBusPriority:
    def test_priority_filter_accepts(self, event_bus):
        received = []
        event_bus.subscribe(
            "test.*",
            lambda e: received.append(e),
            priority_filter=Priority.NAVIGATION,
        )
        event_bus.publish_sync(
            Event(topic="test.a", priority=Priority.EMERGENCY)
        )
        event_bus.publish_sync(
            Event(topic="test.b", priority=Priority.LOW)
        )
        assert len(received) == 1
        assert received[0].priority == Priority.EMERGENCY

    def test_safety_critical_sync_dispatch(self, event_bus):
        """Safety-critical events bypass the queue (dispatched sync)."""
        received = []
        event_bus.subscribe("safety.*", lambda e: received.append(e))
        event = Event(
            topic="safety.hazard",
            priority=Priority.EMERGENCY,
        )
        # Even without starting the dispatch thread, safety events work
        event_bus.publish(event)
        assert len(received) == 1


# ── Async Dispatch ───────────────────────────────────────────────────


class TestEventBusAsync:
    def test_async_dispatch(self, running_event_bus):
        received = []
        barrier = threading.Event()

        def handler(e):
            received.append(e)
            barrier.set()

        running_event_bus.subscribe("test.async", handler)
        running_event_bus.publish(Event(topic="test.async"))
        assert barrier.wait(timeout=2.0), "Event not dispatched within 2s"
        assert len(received) == 1

    def test_start_stop_lifecycle(self):
        bus = EventBus(enable_metrics=False)
        assert not bus.is_running
        bus.start()
        assert bus.is_running
        bus.stop()
        assert not bus.is_running

    def test_drain_on_stop(self):
        bus = EventBus(enable_metrics=False)
        received = []
        bus.subscribe("test.*", lambda e: received.append(e))
        # Put events without starting dispatch thread
        for i in range(5):
            bus.publish(Event(topic="test.drain", data={"i": i}))
        # Stop will drain remaining events
        bus.stop()
        assert len(received) == 5


# ── Error Handling ───────────────────────────────────────────────────


class TestEventBusErrors:
    def test_handler_exception_does_not_crash(self, event_bus):
        """A failing handler should not prevent other handlers from running."""
        results = []

        def bad_handler(e):
            raise ValueError("boom")

        def good_handler(e):
            results.append(e)

        event_bus.subscribe("test.error", bad_handler)
        event_bus.subscribe("test.error", good_handler)
        event_bus.publish_sync(Event(topic="test.error"))
        assert len(results) == 1


# ── Rate Limiting ────────────────────────────────────────────────────


class TestEventBusRateLimiting:
    def test_rate_limiting_drops_fast_events(self):
        bus = EventBus(rate_limit_s=0.5, enable_metrics=False)
        received = []
        bus.subscribe("test.*", lambda e: received.append(e))
        bus.publish_sync(Event(topic="test.rate"))
        bus.publish_sync(Event(topic="test.rate"))  # should be dropped
        assert len(received) == 1

    def test_safety_events_bypass_rate_limit(self):
        bus = EventBus(rate_limit_s=10.0, enable_metrics=False)
        received = []
        bus.subscribe("safety.*", lambda e: received.append(e))
        for _ in range(5):
            bus.publish(Event(
                topic="safety.hazard",
                priority=Priority.EMERGENCY,
            ))
        assert len(received) == 5


# ── Concurrent Access ────────────────────────────────────────────────


class TestEventBusConcurrency:
    def test_concurrent_publish_subscribe(self, event_bus):
        """Multiple threads publishing/subscribing should not crash."""
        count = threading.atomic() if hasattr(threading, "atomic") else None
        received = []
        lock = threading.Lock()

        def handler(e):
            with lock:
                received.append(e)

        event_bus.subscribe("concurrent.*", handler)

        def publisher(topic_suffix: int):
            for i in range(50):
                event_bus.publish_sync(
                    Event(topic=f"concurrent.{topic_suffix}", data={"i": i})
                )

        threads = [
            threading.Thread(target=publisher, args=(t,)) for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(received) == 200  # 4 threads × 50 events


# ── Filter Unit Tests ────────────────────────────────────────────────


class TestFilters:
    def test_priority_filter(self):
        f = PriorityFilter(max_priority=Priority.NAVIGATION)
        assert f.accepts(Event(topic="a", priority=Priority.EMERGENCY))
        assert f.accepts(Event(topic="a", priority=Priority.NAVIGATION))
        assert not f.accepts(Event(topic="a", priority=Priority.LOW))

    def test_topic_filter_exact(self):
        f = TopicFilter(patterns=["vision.detection"])
        assert f.accepts(Event(topic="vision.detection"))
        assert not f.accepts(Event(topic="vision.face"))

    def test_topic_filter_wildcard(self):
        f = TopicFilter(patterns=["vision.*"])
        assert f.accepts(Event(topic="vision.detection"))
        assert f.accepts(Event(topic="vision.face"))
        assert not f.accepts(Event(topic="navigation.gps"))

    def test_rate_limiter(self):
        rl = RateLimiter(min_interval_s=0.5)
        e = Event(topic="test.rate")
        assert rl.allows(e)
        assert not rl.allows(e)  # too fast

    def test_rate_limiter_safety_bypass(self):
        rl = RateLimiter(min_interval_s=10.0)
        e = Event(topic="safety.x", priority=Priority.EMERGENCY)
        assert rl.allows(e)
        assert rl.allows(e)  # safety always passes
