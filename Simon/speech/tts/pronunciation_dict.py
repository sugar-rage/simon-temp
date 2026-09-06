"""
Custom pronunciation dictionary for TTS.

Replaces domain-specific terms with phonetic spellings that the TTS engine
is more likely to pronounce correctly.
"""

from __future__ import annotations

import re
from typing import Dict


class PronunciationDictionary:
    """Applies phonetic substitutions for better TTS rendering."""

    def __init__(self, overrides: Dict[str, str] = None):
        self._overrides = overrides or {
            "Sriracha": "Sir-rotch-ah",
            "Nvidia": "En-vid-ee-uh",
            "Linux": "Lih-nucks",
            "Ubuntu": "Oo-boon-too",
            "Huawei": "Wah-way",
            "Xiaomi": "Shyow-mee",
        }
        
        # Build regexes for exact word boundary matching
        self._compiled_rules = [
            (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), phonetic)
            for word, phonetic in self._overrides.items()
        ]

    def add_rule(self, word: str, phonetic_spelling: str) -> None:
        """Add a custom pronunciation rule at runtime."""
        self._overrides[word] = phonetic_spelling
        self._compiled_rules.append(
            (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), phonetic_spelling)
        )

    def apply(self, text: str) -> str:
        """Apply pronunciation overrides to the text."""
        if not text:
            return ""
            
        corrected = text
        for pattern, phonetic in self._compiled_rules:
            # Replaces the matched word with its phonetic spelling
            corrected = pattern.sub(phonetic, corrected)
            
        return corrected
