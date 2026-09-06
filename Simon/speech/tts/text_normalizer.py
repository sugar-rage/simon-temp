"""
Text Normalizer for TTS.

Converts abbreviations, numbers, dates, and common symbols into their
spoken equivalents before synthesis. This improves TTS prosody and
prevents engines from stumbling over formatting.
"""

from __future__ import annotations

import re


class TextNormalizer:
    """Normalizes raw text into spoken forms."""

    def __init__(self):
        # Basic abbreviation expansion
        self._abbreviations = {
            r"\bETA\b": "E T A",
            r"\bVIT\b": "V I T",
            r"\bDr\.\b": "Doctor",
            r"\bMr\.\b": "Mister",
            r"\bMrs\.\b": "Missus",
            r"\bSt\.\b": "Street",
            r"\bRd\.\b": "Road",
            r"\bAve\.\b": "Avenue",
            r"\bkm\b": "kilometers",
            r"\bmi\b": "miles",
            r"\bmin\b": "minutes",
            r"\bsec\b": "seconds",
            r"\bhr\b": "hours",
            r"&": "and",
        }
        
        # Pre-compile regexes
        self._compiled_abbrs = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self._abbreviations.items()
        ]

    def normalize(self, text: str) -> str:
        """Apply all normalization rules to the text."""
        if not text:
            return ""
            
        normalized = text
        
        # Expand abbreviations
        for pattern, replacement in self._compiled_abbrs:
            normalized = pattern.sub(replacement, normalized)
            
        # Optional: Add more complex rules (number spelling, date formatting)
        # using libraries like `num2words` or custom regex logic.
        
        # Cleanup extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized
