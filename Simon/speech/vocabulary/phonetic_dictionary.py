"""
Phonetic Dictionary — multi-pronunciation lookup for STT error correction.

Maps common misrecognitions to their intended words using phonetic
similarity.  For example, Whisper may hear "vit" when the user says
"VIT" (a university), or "navigate" may be misrecognized as "novice gate".

Uses a simple edit-distance approach rather than a full phonetic model
(Soundex/Metaphone) to keep dependencies minimal, with the option to
extend to phonetic encodings later.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PhoneticDictionary:
    """Multi-pronunciation phonetic lookup dictionary.

    Stores a mapping of canonical terms to their known phonetic
    variations and common misrecognitions.

    Args:
        entries: Initial dictionary of ``{canonical: [variants]}``.
    """

    def __init__(self, entries: Optional[Dict[str, List[str]]] = None):
        # canonical_lower → (canonical, set of variant_lower)
        self._entries: Dict[str, Tuple[str, Set[str]]] = {}

        if entries:
            for canonical, variants in entries.items():
                self.add(canonical, variants)

        # Seed with common SIMON misrecognitions
        self._seed_defaults()

    def add(self, canonical: str, variants: List[str]) -> None:
        """Add or extend a dictionary entry.

        Args:
            canonical: The correct form (e.g. "VIT University").
            variants:  Known misrecognitions or phonetic variants.
        """
        key = canonical.lower()
        if key in self._entries:
            existing_canonical, existing_variants = self._entries[key]
            existing_variants.update(v.lower() for v in variants)
        else:
            self._entries[key] = (canonical, {v.lower() for v in variants})

    def lookup(self, text: str) -> Optional[str]:
        """Look up a potential misrecognition.

        Args:
            text: The recognized text (possibly incorrect).

        Returns:
            The canonical form if *text* matches a variant, else ``None``.
        """
        lower = text.strip().lower()

        # Direct match against variants
        for key, (canonical, variants) in self._entries.items():
            if lower in variants or lower == key:
                return canonical

        return None

    def correct_text(self, text: str) -> str:
        """Attempt to correct all words in a text string.

        Replaces individual words that match known variants with
        their canonical forms.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text (may be identical if no corrections apply).
        """
        words = text.split()
        corrected = []
        any_changed = False

        for word in words:
            replacement = self.lookup(word)
            if replacement is not None and replacement.lower() != word.lower():
                corrected.append(replacement)
                any_changed = True
                logger.debug(f"Phonetic correction: '{word}' → '{replacement}'")
            else:
                corrected.append(word)

        return " ".join(corrected) if any_changed else text

    def get_all_entries(self) -> Dict[str, List[str]]:
        """Return all entries as ``{canonical: [variants]}``."""
        return {
            canonical: sorted(variants)
            for canonical, variants in self._entries.values()
        }

    @property
    def size(self) -> int:
        """Number of canonical entries."""
        return len(self._entries)

    def _seed_defaults(self) -> None:
        """Seed with common SIMON-specific misrecognitions."""
        defaults = {
            "navigate": ["novice gate", "navigate", "never gate"],
            "describe": ["this tribe", "disk ripe"],
            "battery": ["butterfly", "better he"],
            "Simon": ["semen", "seaman", "salmon", "saimon", "simone"],
            "read text": ["red text", "reed text", "read test"],
            "stop": ["stock", "stalk", "stuff"],
            "help": ["health", "helped", "held"],
            "camera": ["canberra", "camera"],
            "volume": ["wall you", "wall him"],
        }
        for canonical, variants in defaults.items():
            self.add(canonical, variants)
