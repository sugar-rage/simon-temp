"""
Grammar Corrector — applies lightweight grammar rules to STT output.

Fixes common Whisper grammar errors that affect command parsing:
- Article duplication ("the the library" → "the library")
- Preposition errors ("navigate at library" → "navigate to library")
- Contraction expansion for robustness

This is NOT a full grammar checker — it targets only patterns that
cause command parsing failures.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Duplicate word removal pattern
_DUPLICATE_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

# Preposition corrections specific to SIMON commands
_PREPOSITION_FIXES: List[Tuple[re.Pattern, str]] = [
    # "navigate at X" → "navigate to X"
    (re.compile(r"\bnavigate\s+(?:at|in|on)\b", re.IGNORECASE), "navigate to"),
    # "go at X" → "go to X"
    (re.compile(r"\bgo\s+(?:at|in|on)\b", re.IGNORECASE), "go to"),
    # "take me at X" → "take me to X"
    (re.compile(r"\btake\s+me\s+(?:at|in|on)\b", re.IGNORECASE), "take me to"),
    # "read the text" → "read text" (for command matching simplicity)
    (re.compile(r"\bread\s+the\s+text\b", re.IGNORECASE), "read text"),
    # "describe the scene" → "describe scene"
    (re.compile(r"\bdescribe\s+the\s+scene\b", re.IGNORECASE), "describe scene"),
]

# Common contraction expansions that Whisper may produce inconsistently
_CONTRACTIONS: Dict[str, str] = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "what's": "what is",
    "where's": "where is",
}


class GrammarCorrector:
    """Applies lightweight grammar corrections to STT output.

    Args:
        expand_contractions: If True, expand contractions.
    """

    def __init__(self, expand_contractions: bool = False):
        self._expand_contractions = expand_contractions

    def correct(self, text: str) -> str:
        """Apply grammar corrections to text.

        Args:
            text: Input text (typically already lowercased by CommandNormalizer).

        Returns:
            Corrected text.
        """
        result = text

        # 1. Remove duplicate words ("the the" → "the")
        result = _DUPLICATE_WORD.sub(r"\1", result)

        # 2. Fix prepositions
        for pattern, replacement in _PREPOSITION_FIXES:
            result = pattern.sub(replacement, result)

        # 3. Expand contractions (optional)
        if self._expand_contractions:
            for contraction, expansion in _CONTRACTIONS.items():
                result = result.replace(contraction, expansion)

        # 4. Collapse whitespace
        result = re.sub(r"\s+", " ", result).strip()

        if result != text:
            logger.debug(f"Grammar corrected: '{text}' → '{result}'")

        return result
