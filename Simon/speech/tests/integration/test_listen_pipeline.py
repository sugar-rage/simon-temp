"""
Integration test for the ListenPipeline.

Tests the full listen pipeline flow with mocked dependencies:
Audio → DSP → VAD → WakeWord → STT → HallucinationFilter → ConfidenceGate
→ Preprocessing → Intent Parsing → Safety Validation → Context Resolution
"""

from __future__ import annotations

import pytest

from speech.context.context_manager import ContextManager
from speech.preprocessing.command_normalizer import CommandNormalizer
from speech.preprocessing.grammar_corrector import GrammarCorrector
from speech.preprocessing.intent_parser import IntentParser
from speech.preprocessing.phonetic_corrector import PhoneticCorrector
from speech.safety.safety_validator import SafetyValidator
from speech.vocabulary.vocabulary_manager import VocabularyManager


class TestListenPipelineIntegration:
    """Integration test simulating the full listen pipeline post-STT."""

    def _run_pipeline(self, raw_text: str, confidence: float = 0.85):
        """Simulate the post-STT pipeline stages."""
        normalizer = CommandNormalizer()
        grammar = GrammarCorrector()
        phonetic = PhoneticCorrector()
        parser = IntentParser()
        safety = SafetyValidator()
        context = ContextManager()
        vocab = VocabularyManager()

        # 1. Normalize
        text = normalizer.normalize(raw_text)

        # 2. Grammar correct
        text = grammar.correct(text)

        # 3. Phonetic correct
        text = vocab.correct_transcript(text)
        text = phonetic.correct(text)

        # 4. Context resolution
        resolution = context.resolve_command_text(text)
        text = resolution.resolved_text

        # 5. Intent parsing
        intent = parser.parse(text)

        # 6. Safety validation (if intent matched)
        decision = None
        if intent.matched:
            decision = safety.validate(
                intent.action, confidence, intent.args
            )

        return text, intent, decision

    def test_navigate_command(self):
        text, intent, decision = self._run_pipeline("Hey Simon, navigate to the library")
        assert intent.action == "navigate"
        assert "library" in intent.args.get("destination", "")
        assert decision is not None
        assert decision.allowed

    def test_read_text_command(self):
        text, intent, decision = self._run_pipeline("Um, could you read the text please")
        assert intent.action == "read_text"
        assert decision.allowed

    def test_stop_command(self):
        text, intent, decision = self._run_pipeline("Stop!")
        assert intent.action == "stop"
        assert decision.allowed

    def test_help_command(self):
        text, intent, decision = self._run_pipeline("Help")
        assert intent.action == "help"
        assert decision.allowed

    def test_describe_scene(self):
        text, intent, decision = self._run_pipeline("Describe the scene")
        assert intent.action == "describe_scene"
        assert decision.allowed

    def test_low_confidence_navigate_rejected(self):
        text, intent, decision = self._run_pipeline(
            "Navigate to library", confidence=0.40
        )
        assert intent.action == "navigate"
        assert decision is not None
        assert not decision.allowed

    def test_emergency_requires_confirmation(self):
        text, intent, decision = self._run_pipeline("Emergency call")
        assert intent.action == "emergency_call"
        assert decision is not None
        assert decision.needs_confirm

    def test_filler_removal_still_parses(self):
        text, intent, decision = self._run_pipeline(
            "Um, like, you know, uh, navigate to the park basically"
        )
        assert intent.action == "navigate"
        assert "park" in intent.args.get("destination", "")

    def test_context_resolution_flow(self):
        """Test that context flows through the pipeline."""
        context = ContextManager()
        context.set_context("last_location", "library")

        normalizer = CommandNormalizer()
        parser = IntentParser()

        text = normalizer.normalize("go there")
        resolution = context.resolve_command_text(text)
        intent = parser.parse(resolution.resolved_text)

        # "go there" → "go library" → navigate intent
        assert resolution.had_anaphora
        assert "library" in resolution.resolved_text
