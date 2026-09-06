"""
Accent Adapter — accent-specific Whisper prompt tuning.

Uses the user's accent hint (from their profile) to prepend contextual
English text that biases Whisper's language model toward the expected
accent patterns.  This improves recognition accuracy for non-native
English speakers.

Whisper's ``initial_prompt`` mechanism conditions the decoder on
previously "generated" text.  By including accent-representative
phrases, we bias the model toward that accent's phonetic patterns.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Accent-specific Whisper prompt prefixes.
# These phrases prime the decoder for the accent's pronunciation patterns.
_ACCENT_PROMPTS: Dict[str, str] = {
    "indian english": (
        "The following is a conversation in Indian English. "
        "Common words include navigate, describe, battery, volume, camera."
    ),
    "british english": (
        "The following is a conversation in British English. "
        "Common words include navigate, describe, battery, volume, camera."
    ),
    "american english": (
        "The following is a conversation in American English. "
        "Common words include navigate, describe, battery, volume, camera."
    ),
    "australian english": (
        "The following is a conversation in Australian English. "
        "Common words include navigate, describe, battery, volume, camera."
    ),
    "south african english": (
        "The following is a conversation in South African English. "
        "Common words include navigate, describe, battery, volume, camera."
    ),
}


class AccentAdapter:
    """Produces accent-specific prompts for Whisper.

    Args:
        accent_hint:     Accent identifier (e.g. "Indian English").
        custom_prompts:  Additional ``{accent: prompt}`` mappings.
    """

    def __init__(
        self,
        accent_hint: str = "",
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        self._accent = accent_hint.strip().lower()
        self._prompts = dict(_ACCENT_PROMPTS)
        if custom_prompts:
            self._prompts.update(
                {k.lower(): v for k, v in custom_prompts.items()}
            )

    @property
    def accent(self) -> str:
        return self._accent

    @accent.setter
    def accent(self, value: str) -> None:
        self._accent = value.strip().lower()

    def get_accent_prompt(self) -> str:
        """Get the accent-specific prompt prefix.

        Returns:
            Prompt string for the current accent, or empty string if
            no accent is set or not recognized.
        """
        if not self._accent:
            return ""

        prompt = self._prompts.get(self._accent, "")
        if prompt:
            logger.debug(f"Accent prompt for '{self._accent}': {len(prompt)} chars")
        return prompt

    def build_combined_prompt(self, vocab_prompt: str) -> str:
        """Combine accent prompt with vocabulary prompt.

        Args:
            vocab_prompt: The vocabulary-based prompt from VocabularyManager.

        Returns:
            Combined prompt (accent prefix + vocab terms).
        """
        accent_part = self.get_accent_prompt()

        if accent_part and vocab_prompt:
            return f"{accent_part} {vocab_prompt}"
        elif accent_part:
            return accent_part
        else:
            return vocab_prompt

    def add_accent(self, accent: str, prompt: str) -> None:
        """Register a custom accent prompt."""
        self._prompts[accent.lower()] = prompt

    def list_accents(self) -> list[str]:
        """Return all known accent identifiers."""
        return sorted(self._prompts.keys())
