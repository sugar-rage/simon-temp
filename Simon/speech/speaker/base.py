"""
Base Speaker Verifier — abstract interface for speaker verification.

All speaker verifiers implement this ABC so they are interchangeable
via constructor DI.  The SpeechManager selects the appropriate
implementation based on available hardware and model files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VerificationResult:
    """Result of a speaker verification attempt.

    Attributes:
        verified:   True if the speaker is the enrolled user.
        score:      Cosine similarity score (0.0–1.0).
        threshold:  The threshold used for the decision.
        speaker_id: Identified speaker ID (if known).
    """

    verified: bool
    score: float
    threshold: float
    speaker_id: str = ""


class BaseSpeakerVerifier(ABC):
    """Abstract base class for speaker verification engines.

    Implementations must support:
    1. Enrollment: register a speaker from audio samples
    2. Verification: check if new audio matches the enrolled speaker
    3. Embedding extraction: produce a speaker embedding vector
    """

    @abstractmethod
    def enroll(self, audio: np.ndarray, sample_rate: int, speaker_id: str) -> bool:
        """Enroll a speaker from audio samples.

        Args:
            audio:       Audio data as float32 array.
            sample_rate: Audio sample rate.
            speaker_id:  Identifier for the enrolled speaker.

        Returns:
            True if enrollment was successful.
        """
        ...

    @abstractmethod
    def verify(
        self, audio: np.ndarray, sample_rate: int, speaker_id: str
    ) -> VerificationResult:
        """Verify if audio matches the enrolled speaker.

        Args:
            audio:       Audio data as float32 array.
            sample_rate: Audio sample rate.
            speaker_id:  The enrolled speaker to verify against.

        Returns:
            A VerificationResult with the decision and score.
        """
        ...

    @abstractmethod
    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[np.ndarray]:
        """Extract a speaker embedding vector from audio.

        Args:
            audio:       Audio data as float32 array.
            sample_rate: Audio sample rate.

        Returns:
            Embedding vector (typically 192-dim for ECAPA-TDNN), or None on failure.
        """
        ...

    @abstractmethod
    def is_enrolled(self, speaker_id: str) -> bool:
        """Check if a speaker is enrolled."""
        ...

    @abstractmethod
    def remove_enrollment(self, speaker_id: str) -> bool:
        """Remove a speaker enrollment. Returns True if existed."""
        ...
