"""InsightFace-based face recognizer.

Refactored from ``perception/face_recognizer.py``. Uses ArcFace embeddings
for recognition with a folder-based face database.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from core.config.system_config import FaceConfig
from core.errors.exceptions import FaceRecognitionError
from vision.face.base import BaseFaceRecognizer, FaceResult

logger = logging.getLogger("simon.vision.face")


class InsightFaceRecognizer(BaseFaceRecognizer):
    """InsightFace-based face detection and recognition.

    Uses antelopev2 (RetinaFace + ArcFace R100) for detection and
    recognition.  Maintains a folder-based face database with multi-embedding
    support per person.

    Parameters
    ----------
    config : FaceConfig
        Face recognition configuration.
    """

    def __init__(self, config: Optional[FaceConfig] = None) -> None:
        self._config = config or FaceConfig()
        self._app: Any = None
        self._ready = False
        self._known_embeddings: dict[str, list[Any]] = {}  # name → embeddings

    def load(self) -> None:
        """Load the InsightFace model and face database."""
        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
        except ImportError:
            raise FaceRecognitionError("insightface or onnxruntime not installed")

        try:
            providers = ort.get_available_providers()
            use_providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in providers
                else ["CPUExecutionProvider"]
            )

            self._app = FaceAnalysis(
                name=self._config.model, providers=use_providers
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._ready = True
            logger.info("InsightFace model loaded: %s", self._config.model)

            self._load_database()
        except Exception as e:
            self._ready = False
            raise FaceRecognitionError(f"Failed to load InsightFace: {e}")

    def detect_faces(self, frame: Any) -> list[FaceResult]:
        if not self._ready or self._app is None:
            return []

        try:
            faces = self._app.get(frame)
        except Exception as e:
            logger.error("Face detection failed: %s", e)
            return []

        results: list[FaceResult] = []

        for face in faces:
            bbox = tuple(int(v) for v in face.bbox)
            # bbox is (x1, y1, x2, y2) from InsightFace
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < 40 or h < 40:  # min face size
                continue

            embedding = face.normed_embedding if hasattr(face, "normed_embedding") else None

            name = "Unknown"
            best_score = 0.0

            if embedding is not None and self._known_embeddings:
                name, best_score = self._match_embedding(embedding)

            results.append(
                FaceResult(
                    name=name,
                    confidence=best_score,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    embedding=embedding,
                )
            )

        return results

    def save_face(self, frame: Any, face: FaceResult, name: str) -> bool:
        if face.embedding is None:
            return False

        db_dir = os.path.join(self._config.database_dir, name)
        os.makedirs(db_dir, exist_ok=True)

        # Add embedding
        if name not in self._known_embeddings:
            self._known_embeddings[name] = []

        embeddings = self._known_embeddings[name]
        if len(embeddings) >= 10:
            logger.info("Max embeddings reached for %s", name)
            return False

        # Check for duplicate
        for existing in embeddings:
            sim = float(np.dot(face.embedding, existing))
            if sim > 0.95:
                logger.debug("Duplicate embedding for %s (sim=%.3f)", name, sim)
                return False

        embeddings.append(face.embedding)

        # Save embedding file
        idx = len(embeddings) - 1
        npy_path = os.path.join(db_dir, f"{idx}.npy")
        np.save(npy_path, face.embedding)
        logger.info("Saved face embedding for %s (%s)", name, npy_path)
        return True

    def is_ready(self) -> bool:
        return self._ready

    def _match_embedding(self, embedding: Any) -> tuple[str, float]:
        """Match an embedding against all known faces."""
        best_name = "Unknown"
        best_score = 0.0

        for name, embeddings in self._known_embeddings.items():
            for known in embeddings:
                score = float(np.dot(embedding, known))
                if score > best_score:
                    best_score = score
                    best_name = name

        threshold = self._config.similarity_threshold
        if best_score < threshold:
            return "Unknown", best_score

        return best_name, best_score

    def _load_database(self) -> None:
        """Load known face embeddings from disk."""
        db_dir = self._config.database_dir
        if not os.path.isdir(db_dir):
            logger.info("Face database directory not found: %s. Creating it.", db_dir)
            os.makedirs(db_dir, exist_ok=True)
            return

        count = 0
        for name in os.listdir(db_dir):
            person_dir = os.path.join(db_dir, name)
            if not os.path.isdir(person_dir):
                continue

            embeddings: list[Any] = []
            for f in sorted(os.listdir(person_dir)):
                if f.endswith(".npy"):
                    try:
                        emb = np.load(os.path.join(person_dir, f))
                        embeddings.append(emb)
                        count += 1
                    except Exception as e:
                        logger.warning("Failed to load %s/%s: %s", name, f, e)

            if embeddings:
                self._known_embeddings[name] = embeddings

        logger.info(
            "Loaded face database: %d people, %d embeddings",
            len(self._known_embeddings),
            count,
        )

    @property
    def known_count(self) -> int:
        return len(self._known_embeddings)
