"""
ECAPA-TDNN Speaker Verifier — speaker verification using ECAPA-TDNN embeddings.

Uses SpeechBrain's pre-trained ECAPA-TDNN model for speaker embedding
extraction and cosine-similarity-based verification.

The model is loaded lazily on first use and cached.  If SpeechBrain
is not installed, the verifier gracefully degrades (verify always
returns unverified with a warning).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from speech.speaker.base import BaseSpeakerVerifier, VerificationResult

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


class EcapaVerifier(BaseSpeakerVerifier):
    """ECAPA-TDNN speaker verifier.

    Args:
        model_source:       SpeechBrain model source string.
        save_dir:           Directory for caching the downloaded model.
        verification_threshold: Cosine similarity threshold (0.0–1.0).
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        save_dir: str = "data/models/ecapa",
        verification_threshold: float = 0.60,
    ):
        self._model_source = model_source
        self._save_dir = save_dir
        self._threshold = verification_threshold
        self._model = None
        self._enrollments: Dict[str, np.ndarray] = {}  # speaker_id → embedding

    def _load_model(self) -> bool:
        """Lazy-load the ECAPA-TDNN model."""
        if self._model is not None:
            return True

        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self._model_source,
                savedir=self._save_dir,
            )
            logger.info("ECAPA-TDNN model loaded successfully")
            return True
        except ImportError:
            logger.warning(
                "SpeechBrain not installed. Speaker verification unavailable. "
                "Install with: pip install speechbrain"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {e}")
            return False

    def enroll(self, audio: np.ndarray, sample_rate: int, speaker_id: str) -> bool:
        """Enroll a speaker by extracting and storing their embedding."""
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return False

        # Average with existing enrollment if present (multi-sample enrollment)
        if speaker_id in self._enrollments:
            existing = self._enrollments[speaker_id]
            self._enrollments[speaker_id] = (existing + embedding) / 2.0
            logger.info(f"Updated enrollment for speaker '{speaker_id}'")
        else:
            self._enrollments[speaker_id] = embedding
            logger.info(f"Enrolled speaker '{speaker_id}'")

        return True

    def verify(
        self, audio: np.ndarray, sample_rate: int, speaker_id: str
    ) -> VerificationResult:
        """Verify if audio matches the enrolled speaker."""
        if speaker_id not in self._enrollments:
            return VerificationResult(
                verified=False, score=0.0, threshold=self._threshold,
                speaker_id=speaker_id,
            )

        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return VerificationResult(
                verified=False, score=0.0, threshold=self._threshold,
                speaker_id=speaker_id,
            )

        enrolled = self._enrollments[speaker_id]
        score = _cosine_similarity(embedding, enrolled)
        verified = score >= self._threshold

        logger.info(
            f"Speaker verification: id={speaker_id}, "
            f"score={score:.3f}, threshold={self._threshold:.3f}, "
            f"{'VERIFIED' if verified else 'REJECTED'}"
        )

        return VerificationResult(
            verified=verified,
            score=score,
            threshold=self._threshold,
            speaker_id=speaker_id,
        )

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int
    ) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio."""
        if not self._load_model():
            return None

        try:
            import torch

            # Ensure float32, mono
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)

            waveform = torch.tensor(audio).unsqueeze(0)
            embedding = self._model.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def is_enrolled(self, speaker_id: str) -> bool:
        return speaker_id in self._enrollments

    def remove_enrollment(self, speaker_id: str) -> bool:
        return self._enrollments.pop(speaker_id, None) is not None
