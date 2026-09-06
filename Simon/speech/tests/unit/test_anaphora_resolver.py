"""
Unit tests for the AnaphoraResolver.

Tests verify:
- "there" resolves to last_location
- "that" resolves to last_object
- "again" resolves to last_action
- Unresolved anaphora detected but not replaced
- No anaphora in plain commands
- Context updates from executed commands
"""

from __future__ import annotations

import pytest

from speech.context.anaphora_resolver import AnaphoraResolver, Resolution
from speech.context.context_store import ContextStore


@pytest.fixture
def store_with_context():
    store = ContextStore()
    store.set("last_location", "library")
    store.set("last_object", "the sign")
    store.set("last_action", "navigate")
    store.set("last_person", "John")
    return store


class TestAnaphoraResolver:
    """Tests for AnaphoraResolver."""

    def test_there_resolves_to_last_location(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("go there")

        assert result.had_anaphora
        assert result.all_resolved
        assert "library" in result.resolved_text
        assert "there" in result.resolutions

    def test_that_resolves_to_last_object(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("read that")

        assert result.had_anaphora
        assert result.all_resolved
        assert "the sign" in result.resolved_text

    def test_again_resolves_to_last_action(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("do again")

        assert result.had_anaphora
        assert "navigate" in result.resolved_text

    def test_him_resolves_to_last_person(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("call him")

        assert result.had_anaphora
        assert "John" in result.resolved_text

    def test_unresolved_anaphora(self):
        """Anaphora without context slot → detected but not replaced."""
        store = ContextStore()  # Empty store
        resolver = AnaphoraResolver(store)
        result = resolver.resolve("go there")

        assert result.had_anaphora
        assert not result.all_resolved
        assert result.resolved_text == "go there"  # Unchanged

    def test_no_anaphora(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("navigate to park")

        assert not result.had_anaphora
        assert result.resolved_text == "navigate to park"

    def test_empty_text(self, store_with_context):
        resolver = AnaphoraResolver(store_with_context)
        result = resolver.resolve("")
        assert not result.had_anaphora

    def test_update_context_from_command_sets_location(self):
        store = ContextStore()
        resolver = AnaphoraResolver(store)

        resolver.update_context_from_command("navigate", {"destination": "park"})

        assert store.get("last_action") == "navigate"
        assert store.get("last_location") == "park"

    def test_update_context_from_command_sets_object(self):
        store = ContextStore()
        resolver = AnaphoraResolver(store)

        resolver.update_context_from_command("read_text", {"target": "sign"})

        assert store.get("last_action") == "read_text"
        assert store.get("last_object") == "sign"

    def test_update_context_sets_person(self):
        store = ContextStore()
        resolver = AnaphoraResolver(store)

        resolver.update_context_from_command("call", {"contact": "Alice"})

        assert store.get("last_person") == "Alice"

    def test_resolution_dataclass(self):
        r = Resolution(
            original_text="go there",
            resolved_text="go library",
            resolutions={"there": "library"},
            had_anaphora=True,
            all_resolved=True,
        )
        assert r.original_text == "go there"
        assert r.resolved_text == "go library"
        assert r.had_anaphora is True
        assert r.all_resolved is True
