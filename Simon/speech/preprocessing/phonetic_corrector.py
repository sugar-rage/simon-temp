"""
Phonetic Corrector — uses the PhoneticDictionary to fix STT misrecognitions.

This module sits in the preprocessing pipeline between the CommandNormalizer
and the IntentParser.  It replaces words that match known phonetic variants
with their canonical forms.

Example: "novice gate to the library" → "navigate to the library"
"""

from __future__ import annotations

import logging
from typing import Optional

from speech.vocabulary.phonetic_dictionary import PhoneticDictionary

logger = logging.getLogger(__name__)


class PhoneticCorrector:
    """Applies phonetic corrections using a PhoneticDictionary.

    Args:
        dictionary: PhoneticDictionary to use for lookups.
    """

    def __init__(self, dictionary: Optional[PhoneticDictionary] = None):
        self._dict = dictionary or PhoneticDictionary()

    @property
    def dictionary(self) -> PhoneticDictionary:
        return self._dict

    def correct(self, text: str) -> str:
        """Apply phonetic corrections to all words in *text*.

        Args:
            text: Normalised command text.

        Returns:
            Text with misrecognized words replaced by canonical forms.
        """
        result = self._dict.correct_text(text)

        if result != text:
            logger.info(f"Phonetic correction: '{text}' → '{result}'")

        return result
