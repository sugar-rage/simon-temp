"""
Speaker Enrollment — voice enrollment and profile management.

Manages the enrollment workflow:
1. Collect N audio samples from the user
2. Extract embeddings from each sample
3. Average embeddings for a robust voiceprint
4. Store the voiceprint and mark the user profile as enrolled

Also handles re-enrollment and voiceprint refresh.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from speech.speaker.base import BaseSpeakerVerifier

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentSession:
    """Tracks an in-progress enrollment.

    Attributes:
        speaker_id:      Who is being enrolled.
        samples_needed:  Total samples required.
        samples_collected: Number of samples so far.
        status:          Current status (collecting, complete, failed).
    """

    speaker_id: str
    samples_needed: int = 3
    samples_collected: int = 0
    status: str = "collecting"  # collecting | complete | failed
    _embeddings: List[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def progress(self) -> float:
        return self.samples_collected / self.samples_needed if self.samples_needed > 0 else 0.0


class EnrollmentManager:
    """Manages speaker enrollment workflows.

    Args:
        verifier:         Speaker verifier for embedding extraction.
        voiceprint_dir:   Directory for persisting voiceprints.
        samples_required: Number of audio samples for enrollment.
    """

    def __init__(
        self,
        verifier: BaseSpeakerVerifier,
        voiceprint_dir: str = "data/voiceprints",
        samples_required: int = 3,
    ):
        self._verifier = verifier
        self._dir = Path(voiceprint_dir)
        self._samples_required = samples_required
        self._active_session: Optional[EnrollmentSession] = None

    def start_enrollment(self, speaker_id: str) -> EnrollmentSession:
        """Start a new enrollment session.

        Args:
            speaker_id: User identifier.

        Returns:
            The new EnrollmentSession.
        """
        self._active_session = EnrollmentSession(
            speaker_id=speaker_id,
            samples_needed=self._samples_required,
        )
        logger.info(
            f"Enrollment started for '{speaker_id}': "
            f"{self._samples_required} samples needed"
        )
        return self._active_session

    def add_sample(
        self, audio: np.ndarray, sample_rate: int
    ) -> EnrollmentSession:
        """Add an audio sample to the active enrollment.

        Args:
            audio:       Audio data as float32 array.
            sample_rate: Audio sample rate.

        Returns:
            Updated EnrollmentSession.

        Raises:
            RuntimeError: If no enrollment is in progress.
        """
        if self._active_session is None:
            raise RuntimeError("No enrollment session in progress")

        session = self._active_session

        embedding = self._verifier.extract_embedding(audio, sample_rate)
        if embedding is None:
            logger.warning("Failed to extract embedding from enrollment sample")
            session.status = "failed"
            return session

        session._embeddings.append(embedding)
        session.samples_collected += 1

        logger.info(
            f"Enrollment sample {session.samples_collected}/{session.samples_needed} "
            f"for '{session.speaker_id}'"
        )

        # Check if enrollment is complete
        if session.samples_collected >= session.samples_needed:
            self._finalize_enrollment(session)

        return session

    def cancel_enrollment(self) -> None:
        """Cancel the active enrollment session."""
        if self._active_session:
            logger.info(f"Enrollment cancelled for '{self._active_session.speaker_id}'")
            self._active_session = None

    @property
    def active_session(self) -> Optional[EnrollmentSession]:
        return self._active_session

    def is_enrolled(self, speaker_id: str) -> bool:
        """Check if a speaker is enrolled (in verifier or on disk)."""
        if self._verifier.is_enrolled(speaker_id):
            return True
        return (self._dir / f"{speaker_id}.npy").exists()

    def load_voiceprint(self, speaker_id: str) -> bool:
        """Load a saved voiceprint into the verifier.

        Returns True if successfully loaded.
        """
        path = self._dir / f"{speaker_id}.npy"
        if not path.exists():
            return False

        try:
            embedding = np.load(str(path))
            # Enroll directly with the loaded embedding
            # We use a synthetic 1-second silent audio + override
            # Actually we just store the embedding in the verifier's dict
            if hasattr(self._verifier, '_enrollments'):
                self._verifier._enrollments[speaker_id] = embedding
                logger.info(f"Loaded voiceprint for '{speaker_id}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load voiceprint for '{speaker_id}': {e}")
            return False

    def _finalize_enrollment(self, session: EnrollmentSession) -> None:
        """Finalize enrollment by averaging embeddings and saving."""
        if not session._embeddings:
            session.status = "failed"
            return

        # Average all embeddings
        avg_embedding = np.mean(session._embeddings, axis=0)

        # Store in verifier
        if hasattr(self._verifier, '_enrollments'):
            self._verifier._enrollments[session.speaker_id] = avg_embedding

        # Persist to disk
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            np.save(str(self._dir / f"{session.speaker_id}.npy"), avg_embedding)
        except OSError as e:
            logger.warning(f"Failed to save voiceprint: {e}")

        session.status = "complete"
        self._active_session = None
        logger.info(f"Enrollment complete for '{session.speaker_id}'")
