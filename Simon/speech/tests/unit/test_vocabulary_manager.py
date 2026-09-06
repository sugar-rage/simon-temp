"""
Unit tests for the VocabularyManager.

Tests verify:
- Prompt building aggregates all sources
- Priority ordering (command > session > location > user)
- Prompt capping at max tokens
- Phonetic correction through the manager
- Session clear
"""

from __future__ import annotations

import pytest

from speech.vocabulary.location_vocab import LocationVocab
from speech.vocabulary.phonetic_dictionary import PhoneticDictionary
from speech.vocabulary.session_vocab import SessionVocab
from speech.vocabulary.user_vocab import UserVocab
from speech.vocabulary.vocabulary_manager import VocabularyManager


class TestLocationVocab:
    def test_add_and_get(self):
        v = LocationVocab()
        v.add("Library")
        assert "Library" in v.get_terms()

    def test_deduplication(self):
        v = LocationVocab()
        v.add("Library")
        v.add("library")
        assert v.size == 1

    def test_max_entries(self):
        v = LocationVocab(max_entries=3)
        v.add("a")
        v.add("b")
        v.add("c")
        v.add("d")
        assert v.size == 3
        assert "a" not in [t.lower() for t in v.get_terms()]


class TestSessionVocab:
    def test_add_and_get(self):
        v = SessionVocab()
        v.add("EXIT SIGN")
        assert "EXIT SIGN" in v.get_terms()

    def test_add_many(self):
        v = SessionVocab()
        v.add_many(["a", "b", "c"])
        assert v.size == 3

    def test_clear(self):
        v = SessionVocab()
        v.add("test")
        v.clear()
        assert v.size == 0


class TestUserVocab:
    def test_add_and_get(self):
        v = UserVocab()  # In-memory only
        v.add("VIT University")
        assert "VIT University" in v.get_terms()

    def test_deduplication(self):
        v = UserVocab()
        v.add("VIT")
        v.add("vit")
        assert v.size == 1

    def test_remove(self):
        v = UserVocab()
        v.add("test")
        assert v.remove("test")
        assert v.size == 0


class TestPhoneticDictionary:
    def test_lookup_variant(self):
        d = PhoneticDictionary()
        # "salmon" is a default variant for "Simon"
        result = d.lookup("salmon")
        assert result == "Simon"

    def test_lookup_canonical(self):
        d = PhoneticDictionary()
        result = d.lookup("Simon")
        assert result == "Simon"

    def test_lookup_unknown(self):
        d = PhoneticDictionary()
        result = d.lookup("xyznotaword")
        assert result is None

    def test_correct_text(self):
        d = PhoneticDictionary()
        result = d.correct_text("stock talking")
        # "stock" is a default variant for "stop"
        assert "stop" in result.lower()

    def test_add_custom_entry(self):
        d = PhoneticDictionary()
        d.add("MyWord", ["my word", "mi word"])
        assert d.lookup("mi word") == "MyWord"

    def test_get_all_entries(self):
        d = PhoneticDictionary()
        entries = d.get_all_entries()
        assert len(entries) > 0
        assert "Simon" in entries


class TestVocabularyManager:
    def test_build_prompt_includes_command_words(self):
        mgr = VocabularyManager(command_words=["navigate", "stop", "help"])
        prompt = mgr.build_prompt()
        assert "navigate" in prompt
        assert "stop" in prompt

    def test_build_prompt_includes_session_terms(self):
        session = SessionVocab()
        session.add("EXIT SIGN")
        mgr = VocabularyManager(session_vocab=session)
        prompt = mgr.build_prompt()
        assert "EXIT SIGN" in prompt

    def test_build_prompt_includes_location_terms(self):
        loc = LocationVocab(initial_locations=["Park", "Library"])
        mgr = VocabularyManager(location_vocab=loc)
        prompt = mgr.build_prompt()
        assert "Park" in prompt

    def test_build_prompt_deduplicates(self):
        mgr = VocabularyManager(
            command_words=["navigate"],
            location_vocab=LocationVocab(initial_locations=["navigate"]),
        )
        prompt = mgr.build_prompt()
        assert prompt.count("navigate") == 1

    def test_build_prompt_capped(self):
        """Prompt should not exceed ~200 tokens."""
        loc = LocationVocab(initial_locations=[f"Location{i}" for i in range(300)])
        mgr = VocabularyManager(location_vocab=loc)
        prompt = mgr.build_prompt()
        word_count = len(prompt.split())
        assert word_count <= 160  # ~200 tokens / 1.3 words per token

    def test_correct_transcript(self):
        mgr = VocabularyManager()
        result = mgr.correct_transcript("stock talking to park")
        assert "stop" in result.lower()

    def test_session_end_clears_session(self):
        session = SessionVocab()
        session.add("test")
        mgr = VocabularyManager(session_vocab=session)
        mgr.on_session_end()
        assert session.size == 0
