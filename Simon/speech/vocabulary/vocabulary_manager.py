"""
Vocabulary Manager — aggregates all vocabulary sources into an STT prompt.

Combines:
- Location vocabulary (recently navigated places)
- User vocabulary (explicitly taught words)
- Session vocabulary (OCR text, object labels)
- Command vocabulary (action names from config)

Produces the ``initial_prompt`` string injected into Whisper's decoder
to bias recognition toward expected words.  The prompt is capped at
~200 tokens to avoid degrading recognition of non-vocabulary words.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from speech.vocabulary.location_vocab import LocationVocab
from speech.vocabulary.phonetic_dictionary import PhoneticDictionary
from speech.vocabulary.session_vocab import SessionVocab
from speech.vocabulary.user_vocab import UserVocab

logger = logging.getLogger(__name__)

# Approximate tokens-per-word ratio for English
_APPROX_TOKENS_PER_WORD = 1.3
_MAX_PROMPT_TOKENS = 200
_MAX_PROMPT_WORDS = int(_MAX_PROMPT_TOKENS / _APPROX_TOKENS_PER_WORD)


class VocabularyManager:
    """Aggregates vocabulary sources and produces STT biasing prompts.

    Args:
        location_vocab: LocationVocab instance.
        user_vocab:     UserVocab instance.
        session_vocab:  SessionVocab instance.
        phonetic_dict:  PhoneticDictionary instance.
        command_words:  Static command action names (e.g. from COMMAND_MAP).
    """

    def __init__(
        self,
        location_vocab: Optional[LocationVocab] = None,
        user_vocab: Optional[UserVocab] = None,
        session_vocab: Optional[SessionVocab] = None,
        phonetic_dict: Optional[PhoneticDictionary] = None,
        command_words: Optional[List[str]] = None,
    ):
        self._location = location_vocab or LocationVocab()
        self._user = user_vocab or UserVocab()
        self._session = session_vocab or SessionVocab()
        self._phonetic = phonetic_dict or PhoneticDictionary()
        self._command_words = command_words or []

    @property
    def location_vocab(self) -> LocationVocab:
        return self._location

    @property
    def user_vocab(self) -> UserVocab:
        return self._user

    @property
    def session_vocab(self) -> SessionVocab:
        return self._session

    @property
    def phonetic_dict(self) -> PhoneticDictionary:
        return self._phonetic

    def build_prompt(self) -> str:
        """Build the STT initial_prompt string from all vocab sources.

        Priority order (highest first):
        1. Command words (most important for action recognition)
        2. Session vocab (recently observed terms)
        3. Location vocab (navigation targets)
        4. User vocab (custom terms)

        Returns:
            A space-separated string of vocabulary terms, capped at ~200 tokens.
        """
        seen: Set[str] = set()
        terms: List[str] = []

        def add_terms(source_terms: List[str]) -> None:
            for term in source_terms:
                lower = term.lower()
                if lower not in seen:
                    seen.add(lower)
                    terms.append(term)

        # Priority 1: Command words
        add_terms(self._command_words)

        # Priority 2: Session vocab (most recent context)
        add_terms(self._session.get_terms())

        # Priority 3: Location vocab
        add_terms(self._location.get_terms())

        # Priority 4: User vocab
        add_terms(self._user.get_terms())

        # Cap at max words
        if len(terms) > _MAX_PROMPT_WORDS:
            terms = terms[:_MAX_PROMPT_WORDS]

        prompt = " ".join(terms)
        logger.debug(
            f"Built STT prompt: {len(terms)} terms, "
            f"~{int(len(terms) * _APPROX_TOKENS_PER_WORD)} tokens"
        )
        return prompt

    def correct_transcript(self, text: str) -> str:
        """Apply phonetic corrections to a transcript.

        Args:
            text: Raw transcript text.

        Returns:
            Corrected text.
        """
        return self._phonetic.correct_text(text)

    def on_session_end(self) -> None:
        """Clear session-scoped vocabulary."""
        self._session.clear()
        logger.debug("Session vocabulary cleared")
