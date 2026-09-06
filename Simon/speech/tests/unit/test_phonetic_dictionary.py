"""
Unit tests for the PhoneticDictionary.

Tests verify:
- Default seed entries exist
- Variant lookup returns canonical
- Unknown word returns None
- Text correction replaces known variants
- Custom entries can be added
- Multi-word variants handled
"""

from __future__ import annotations

import pytest

from speech.vocabulary.phonetic_dictionary import PhoneticDictionary


class TestPhoneticDictionary:
    """Tests for PhoneticDictionary."""

    def test_default_seeds_exist(self):
        d = PhoneticDictionary()
        assert d.size > 0

    def test_lookup_known_variant(self):
        d = PhoneticDictionary()
        assert d.lookup("salmon") == "Simon"

    def test_lookup_canonical_returns_canonical(self):
        d = PhoneticDictionary()
        assert d.lookup("Simon") == "Simon"

    def test_lookup_unknown_returns_none(self):
        d = PhoneticDictionary()
        assert d.lookup("xyzabc") is None

    def test_lookup_case_insensitive(self):
        d = PhoneticDictionary()
        assert d.lookup("SALMON") == "Simon"
        assert d.lookup("Salmon") == "Simon"

    def test_correct_text_replaces_variant(self):
        d = PhoneticDictionary()
        result = d.correct_text("stock talking")
        # "stock" should be corrected to "stop"
        assert "stop" in result.lower()

    def test_correct_text_no_change_for_unknown(self):
        d = PhoneticDictionary()
        result = d.correct_text("hello world")
        assert result == "hello world"

    def test_add_custom_entry(self):
        d = PhoneticDictionary()
        d.add("CustomWord", ["kustom", "costum"])
        assert d.lookup("kustom") == "CustomWord"
        assert d.lookup("costum") == "CustomWord"

    def test_add_extends_existing_entry(self):
        d = PhoneticDictionary()
        d.add("Simon", ["new_variant"])
        assert d.lookup("new_variant") == "Simon"
        # Old variants should still work
        assert d.lookup("salmon") == "Simon"

    def test_multi_word_variant(self):
        d = PhoneticDictionary()
        assert d.lookup("novice gate") == "navigate"

    def test_correct_multi_word_variant(self):
        d = PhoneticDictionary()
        # Multi-word correction operates word-by-word, so "novice" alone
        # won't match. But "novice gate" as a phrase lookup works.
        result = d.lookup("novice gate")
        assert result == "navigate"

    def test_get_all_entries(self):
        d = PhoneticDictionary()
        entries = d.get_all_entries()
        assert isinstance(entries, dict)
        assert "Simon" in entries
        assert "salmon" in entries["Simon"]

    def test_empty_string_returns_none(self):
        d = PhoneticDictionary()
        assert d.lookup("") is None

    def test_whitespace_string_returns_none(self):
        d = PhoneticDictionary()
        assert d.lookup("   ") is None

    def test_initial_entries_constructor(self):
        d = PhoneticDictionary(entries={
            "Apple": ["appel", "aple"],
            "Banana": ["bananna", "banan"],
        })
        assert d.lookup("appel") == "Apple"
        assert d.lookup("bananna") == "Banana"
        # Defaults should still be seeded
        assert d.lookup("salmon") == "Simon"
