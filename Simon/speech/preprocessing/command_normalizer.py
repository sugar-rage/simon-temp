"""
Command Normalizer — normalizes raw STT transcript text for command parsing.

Handles:
- Filler word removal ("um", "uh", "like", "you know")
- Verb normalization ("navigating" → "navigate")
- Whitespace/punctuation cleanup
- Case normalization
"""

from __future__ import annotations

import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)

# Filler words and discourse markers to remove
_FILLERS: Set[str] = {
    "um", "uh", "uhm", "umm", "er", "ah", "hmm",
    "like", "you know", "basically", "actually",
    "so", "well", "okay so", "right",
    "please", "can you", "could you", "would you",
    "i want to", "i need to", "i'd like to",
    "hey simon", "hey", "simon",
}

# Common verb normalizations (present participle → base form)
_VERB_NORMALIZATIONS = {
    "navigating": "navigate",
    "reading": "read",
    "describing": "describe",
    "stopping": "stop",
    "calling": "call",
    "helping": "help",
    "taking": "take",
    "opening": "open",
    "closing": "close",
    "starting": "start",
    "scanning": "scan",
    "detecting": "detect",
}


class CommandNormalizer:
    """Normalizes raw transcript text for command parsing.

    Args:
        extra_fillers:          Additional filler words to remove.
        extra_normalizations:   Additional verb normalizations.
    """

    def __init__(
        self,
        extra_fillers: List[str] | None = None,
        extra_normalizations: dict[str, str] | None = None,
    ):
        self._fillers = set(_FILLERS)
        if extra_fillers:
            self._fillers.update(f.lower() for f in extra_fillers)

        self._normalizations = dict(_VERB_NORMALIZATIONS)
        if extra_normalizations:
            self._normalizations.update(
                {k.lower(): v for k, v in extra_normalizations.items()}
            )

        # Sort fillers by length (longest first) for greedy matching
        self._filler_patterns = sorted(self._fillers, key=len, reverse=True)

    def normalize(self, text: str) -> str:
        """Normalize a raw transcript for command parsing.

        Args:
            text: Raw STT text.

        Returns:
            Normalized text with fillers removed and verbs normalized.
        """
        result = text.strip()

        # 1. Lowercase
        result = result.lower()

        # 2. Remove punctuation (keep apostrophes for contractions)
        result = re.sub(r"[^\w\s']", " ", result)

        # 3. Remove fillers (multi-word first, then single-word)
        for filler in self._filler_patterns:
            # Use word boundaries for single words, exact match for multi-word
            if " " in filler:
                result = result.replace(filler, " ")
            else:
                result = re.sub(rf"\b{re.escape(filler)}\b", " ", result)

        # 4. Normalize verbs
        words = result.split()
        normalized_words = []
        for word in words:
            normalized_words.append(self._normalizations.get(word, word))
        result = " ".join(normalized_words)

        # 5. Collapse whitespace
        result = re.sub(r"\s+", " ", result).strip()

        if result != text.strip().lower():
            logger.debug(f"Command normalized: '{text}' → '{result}'")

        return result
