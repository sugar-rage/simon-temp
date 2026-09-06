"""Unit tests for the CapabilityRegistry — registration, queries, listeners."""

from __future__ import annotations

import threading

import pytest

from core.capabilities.registry import CapabilityRegistry, CapabilityStatus


class TestCapabilityRegistration:
    def test_register_available(self, capability_registry):
        capability_registry.register("capability.camera", available=True)
        assert capability_registry.is_available("capability.camera")

    def test_register_unavailable(self, capability_registry):
        capability_registry.register(
            "capability.ocr",
            available=False,
            reason="not installed",
        )
        assert not capability_registry.is_available("capability.ocr")

    def test_register_degraded(self, capability_registry):
        capability_registry.register(
            "capability.detection",
            available=True,
            degraded=True,
        )
        assert capability_registry.is_available("capability.detection")
        assert capability_registry.is_degraded("capability.detection")

    def test_register_with_metadata(self, capability_registry):
        capability_registry.register(
            "capability.camera",
            available=True,
            metadata={"device": "/dev/video0"},
        )
        status = capability_registry.get_status("capability.camera")
        assert status.metadata["device"] == "/dev/video0"

    def test_unknown_capability_not_available(self, capability_registry):
        assert not capability_registry.is_available("capability.unknown")

    def test_unknown_capability_status_is_none(self, capability_registry):
        assert capability_registry.get_status("capability.unknown") is None


class TestCapabilityUpdate:
    def test_update_existing(self, capability_registry):
        capability_registry.register("capability.gps", available=True)
        capability_registry.update("capability.gps", available=False, reason="signal lost")
        assert not capability_registry.is_available("capability.gps")
        status = capability_registry.get_status("capability.gps")
        assert status.reason == "signal lost"

    def test_update_preserves_unset_fields(self, capability_registry):
        capability_registry.register(
            "capability.camera",
            available=True,
            metadata={"device": "cam0"},
        )
        capability_registry.update("capability.camera", degraded=True)
        status = capability_registry.get_status("capability.camera")
        assert status.available  # preserved
        assert status.degraded  # updated
        assert status.metadata["device"] == "cam0"  # preserved

    def test_update_nonexistent_creates(self, capability_registry):
        capability_registry.update("capability.new", available=True)
        assert capability_registry.is_available("capability.new")


class TestCapabilityListing:
    def test_list_available(self, capability_registry):
        capability_registry.register("capability.a", available=True)
        capability_registry.register("capability.b", available=False)
        capability_registry.register("capability.c", available=True)
        available = capability_registry.list_available()
        assert set(available) == {"capability.a", "capability.c"}

    def test_list_degraded(self, capability_registry):
        capability_registry.register("capability.a", available=True, degraded=True)
        capability_registry.register("capability.b", available=True)
        assert capability_registry.list_degraded() == ["capability.a"]

    def test_list_unavailable(self, capability_registry):
        capability_registry.register("capability.a", available=False)
        capability_registry.register("capability.b", available=True)
        assert capability_registry.list_unavailable() == ["capability.a"]

    def test_list_all(self, capability_registry):
        capability_registry.register("capability.x", available=True)
        capability_registry.register("capability.y", available=False)
        all_caps = capability_registry.list_all()
        assert len(all_caps) == 2

    def test_degraded_excluded_from_available(self, capability_registry):
        """Degraded capabilities should NOT appear in list_available()."""
        capability_registry.register(
            "capability.ocr", available=True, degraded=True
        )
        assert "capability.ocr" not in capability_registry.list_available()


class TestCapabilityListeners:
    def test_listener_called_on_register(self, capability_registry):
        changes = []
        capability_registry.subscribe_changes(
            lambda name, status: changes.append((name, status.available))
        )
        capability_registry.register("capability.camera", available=True)
        assert len(changes) == 1
        assert changes[0] == ("capability.camera", True)

    def test_listener_called_on_update(self, capability_registry):
        changes = []
        capability_registry.register("capability.gps", available=True)
        capability_registry.subscribe_changes(
            lambda name, status: changes.append((name, status.available))
        )
        capability_registry.update("capability.gps", available=False)
        assert len(changes) == 1
        assert changes[0] == ("capability.gps", False)

    def test_unsubscribe_listener(self, capability_registry):
        changes = []
        listener = lambda name, status: changes.append(name)
        capability_registry.subscribe_changes(listener)
        capability_registry.unsubscribe_changes(listener)
        capability_registry.register("capability.x", available=True)
        assert len(changes) == 0

    def test_listener_exception_does_not_crash(self, capability_registry):
        """A failing listener should not prevent registration."""
        def bad_listener(name, status):
            raise ValueError("boom")

        capability_registry.subscribe_changes(bad_listener)
        # Should not raise
        capability_registry.register("capability.test", available=True)
        assert capability_registry.is_available("capability.test")


class TestCapabilitySummary:
    def test_summary_empty(self, capability_registry):
        s = capability_registry.summary()
        assert "none registered" in s

    def test_summary_with_capabilities(self, capability_registry):
        capability_registry.register("capability.camera", available=True)
        capability_registry.register(
            "capability.gps", available=False, reason="not found"
        )
        capability_registry.register(
            "capability.ocr", available=True, degraded=True, reason="fallback"
        )
        summary = capability_registry.summary()
        assert "capability.camera" in summary
        assert "capability.gps" in summary
        assert "not found" in summary
        assert "degraded" in summary


class TestCapabilityThreadSafety:
    def test_concurrent_register_and_query(self, capability_registry):
        """Multiple threads registering and querying should not crash."""
        errors = []

        def register_worker(i):
            try:
                capability_registry.register(
                    f"capability.test_{i}", available=True
                )
                capability_registry.is_available(f"capability.test_{i}")
                capability_registry.list_available()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_worker, args=(i,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(capability_registry.list_all()) == 20
