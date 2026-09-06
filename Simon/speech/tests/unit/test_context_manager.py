"""
Unit tests for the ContextManager and ContextStore.

Tests verify:
- Slot set/get with TTL expiry
- Expired slot returns default
- Purge removes expired slots
- Max slots eviction
- Context resolution through the ContextManager
- State summary reflects active slots
"""

from __future__ import annotations

import time

import pytest

from speech.context.context_manager import ContextManager
from speech.context.context_store import ContextSlot, ContextStore


class TestContextStore:
    """Tests for ContextStore."""

    def test_set_and_get(self):
        store = ContextStore()
        store.set("last_location", "library")
        assert store.get("last_location") == "library"

    def test_get_missing_returns_default(self):
        store = ContextStore()
        assert store.get("nonexistent") is None
        assert store.get("nonexistent", "fallback") == "fallback"

    def test_expired_slot_returns_default(self):
        store = ContextStore()
        store.set("temp", "value", ttl_s=0.01)
        time.sleep(0.02)
        assert store.get("temp") is None

    def test_ttl_zero_never_expires(self):
        store = ContextStore()
        store.set("permanent", "value", ttl_s=0)
        # Slot with ttl=0 should never expire
        slot = store.get_slot("permanent")
        assert slot is not None
        assert not slot.is_expired

    def test_remove_slot(self):
        store = ContextStore()
        store.set("key", "val")
        assert store.remove("key") is True
        assert store.get("key") is None
        assert store.remove("key") is False

    def test_clear(self):
        store = ContextStore()
        store.set("a", 1)
        store.set("b", 2)
        store.clear()
        assert store.size == 0

    def test_purge_expired(self):
        store = ContextStore()
        store.set("short", "a", ttl_s=0.01)
        store.set("long", "b", ttl_s=300)
        time.sleep(0.02)

        purged = store.purge_expired()
        assert purged == 1
        assert store.get("short") is None
        assert store.get("long") == "b"

    def test_max_slots_eviction(self):
        store = ContextStore(max_slots=3)
        store.set("a", 1)
        store.set("b", 2)
        store.set("c", 3)
        store.set("d", 4)  # Should evict "a"

        assert store.get("a") is None
        assert store.get("d") == 4

    def test_has_checks_expiry(self):
        store = ContextStore()
        store.set("temp", "val", ttl_s=0.01)
        assert store.has("temp") is True
        time.sleep(0.02)
        assert store.has("temp") is False

    def test_all_active_excludes_expired(self):
        store = ContextStore()
        store.set("live", "yes", ttl_s=300)
        store.set("dead", "no", ttl_s=0.01)
        time.sleep(0.02)

        active = store.all_active()
        assert "live" in active
        assert "dead" not in active

    def test_overwrite_resets_timestamp(self):
        store = ContextStore()
        store.set("key", "v1")
        slot1 = store.get_slot("key")
        ts1 = slot1.timestamp

        time.sleep(0.02)
        store.set("key", "v2")
        slot2 = store.get_slot("key")

        assert slot2.value == "v2"
        assert slot2.timestamp >= ts1


class TestContextSlot:
    """Tests for ContextSlot."""

    def test_is_expired(self):
        slot = ContextSlot(name="test", value="x", ttl_s=0.01)
        assert not slot.is_expired
        time.sleep(0.02)
        assert slot.is_expired

    def test_age_increases(self):
        slot = ContextSlot(name="test", value="x")
        time.sleep(0.025)
        assert slot.age_s >= 0


class TestContextManager:
    """Tests for ContextManager."""

    def test_set_and_get_context(self):
        mgr = ContextManager()
        mgr.set_context("last_location", "library")
        assert mgr.get_context("last_location") == "library"

    def test_resolve_command_with_no_anaphora(self):
        mgr = ContextManager()
        resolution = mgr.resolve_command_text("navigate to library")
        assert not resolution.had_anaphora
        assert resolution.resolved_text == "navigate to library"

    def test_resolve_command_with_anaphora(self):
        mgr = ContextManager()
        mgr.set_context("last_location", "library")
        resolution = mgr.resolve_command_text("go there")
        assert resolution.had_anaphora
        assert "library" in resolution.resolved_text

    def test_update_after_command(self):
        mgr = ContextManager()
        mgr.update_after_command("navigate", {"destination": "park"})
        assert mgr.get_context("last_location") == "park"
        assert mgr.get_context("last_action") == "navigate"

    def test_state_summary(self):
        mgr = ContextManager()
        mgr.set_context("a", 1)
        mgr.set_context("b", 2)
        summary = mgr.get_state_summary()
        assert "a" in summary
        assert "b" in summary

    def test_clear(self):
        mgr = ContextManager()
        mgr.set_context("key", "val")
        mgr.clear()
        assert mgr.get_context("key") is None
