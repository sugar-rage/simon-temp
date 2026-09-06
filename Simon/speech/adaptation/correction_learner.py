"""
Correction Learner — learns from user corrections to improve recognition.

When a user explicitly corrects a misrecognition (e.g. "I said 'navigate'
not 'novice gate'"), this module:
1. Records the correction in the user profile
2. Updates the phonetic dictionary with the new mapping
3. Updates the vocabulary manager with the corrected term

Over time this builds a personalized error correction model.
"""

from __future__ import annotations

import logging
from typing import Optional

from speech.adaptation.user_profile import UserProfile
from speech.vocabulary.phonetic_dictionary import PhoneticDictionary

logger = logging.getLogger(__name__)


class CorrectionLearner:
    """Learns from explicit user corrections.

    Args:
        profile:         Active user profile for persistence.
        phonetic_dict:   PhoneticDictionary to extend with corrections.
        max_corrections: Maximum corrections to store per user.
    """

    def __init__(
        self,
        profile: Optional[UserProfile] = None,
        phonetic_dict: Optional[PhoneticDictionary] = None,
        max_corrections: int = 500,
    ):
        self._profile = profile
        self._phonetic = phonetic_dict
        self._max = max_corrections
        self._session_corrections: dict[str, str] = {}

        # Replay existing corrections into phonetic dict
        if self._profile and self._phonetic:
            for wrong, right in self._profile.corrections.items():
                self._phonetic.add(right, [wrong])

    def learn(self, wrong: str, right: str) -> None:
        """Record a user correction.

        Args:
            wrong: The misrecognized text.
            right: The intended text.
        """
        wrong_lower = wrong.strip().lower()
        right_clean = right.strip()

        if not wrong_lower or not right_clean:
            return

        # 1. Record in session memory
        self._session_corrections[wrong_lower] = right_clean

        # 2. Record in user profile (persistent)
        if self._profile:
            self._profile.add_correction(wrong_lower, right_clean)

            # Cap corrections
            if len(self._profile.corrections) > self._max:
                # Remove oldest (first inserted)
                oldest_key = next(iter(self._profile.corrections))
                del self._profile.corrections[oldest_key]

        # 3. Update phonetic dictionary
        if self._phonetic:
            self._phonetic.add(right_clean, [wrong_lower])

        logger.info(f"Correction learned: '{wrong}' → '{right_clean}'")

    def apply_corrections(self, text: str) -> str:
        """Apply known corrections to text.

        Checks session corrections first (most recent), then profile.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        lower = text.strip().lower()

        # Session corrections (most recent)
        if lower in self._session_corrections:
            corrected = self._session_corrections[lower]
            logger.debug(f"Session correction applied: '{text}' → '{corrected}'")
            return corrected

        # Profile corrections (persistent)
        if self._profile:
            corrected = self._profile.get_correction(lower)
            if corrected:
                logger.debug(f"Profile correction applied: '{text}' → '{corrected}'")
                return corrected

        return text

    @property
    def correction_count(self) -> int:
        """Total unique corrections known."""
        count = len(self._session_corrections)
        if self._profile:
            count += len(self._profile.corrections)
        return count
