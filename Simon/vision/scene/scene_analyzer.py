"""Scene analyzer — classifies the overall scene from frame content.

Uses the distribution of YOLO detections to classify the scene as
indoor, outdoor, street, park, etc.  Runs periodically (every N frames)
to avoid per-frame overhead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.models.detection import Detection

logger = logging.getLogger("simon.vision.scene")


@dataclass
class SceneAnalysis:
    """Result of scene classification.

    Attributes
    ----------
    scene_type : str
        Scene classification (e.g. ``"street"``, ``"indoor"``, ``"park"``).
    confidence : float
        Classification confidence.
    description : str
        Human-readable scene description.
    object_counts : dict[str, int]
        Count of each object class in the scene.
    """

    scene_type: str = "unknown"
    confidence: float = 0.0
    description: str = ""
    object_counts: dict[str, int] | None = None


# Object classes that suggest specific scene types
_STREET_INDICATORS = {"car", "truck", "bus", "motorcycle", "traffic light", "stop sign"}
_INDOOR_INDICATORS = {"chair", "couch", "bed", "tv", "laptop", "dining table"}
_PARK_INDICATORS = {"dog", "cat", "bird", "bench", "potted plant"}
_PERSON_CLASSES = {"person"}


class SceneAnalyzer:
    """Detection-based scene classifier.

    Analyzes the distribution of detected objects to classify the scene.
    Lightweight — no additional model required.
    """

    def analyze(self, detections: list[Detection]) -> SceneAnalysis:
        """Classify the scene based on current detections.

        Parameters
        ----------
        detections : list[Detection]
            Current frame detections.

        Returns
        -------
        SceneAnalysis
        """
        if not detections:
            return SceneAnalysis(
                scene_type="unknown",
                confidence=0.0,
                description="No objects detected",
                object_counts={},
            )

        # Count object classes
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.cls] = counts.get(det.cls, 0) + 1

        # Score each scene type
        street_score = sum(counts.get(c, 0) for c in _STREET_INDICATORS)
        indoor_score = sum(counts.get(c, 0) for c in _INDOOR_INDICATORS)
        park_score = sum(counts.get(c, 0) for c in _PARK_INDICATORS)
        person_count = sum(counts.get(c, 0) for c in _PERSON_CLASSES)

        total = len(detections)
        scores = {
            "street": street_score,
            "indoor": indoor_score,
            "park": park_score,
        }

        best_scene = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_scene]

        if best_score == 0:
            scene_type = "outdoor" if person_count > 0 else "unknown"
            confidence = 0.3
        else:
            scene_type = best_scene
            confidence = min(best_score / max(total, 1), 1.0)

        # Build description
        top_objects = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        desc_parts = [f"{count} {cls}" for cls, count in top_objects]
        description = f"{scene_type.title()} scene with {', '.join(desc_parts)}"

        return SceneAnalysis(
            scene_type=scene_type,
            confidence=confidence,
            description=description,
            object_counts=counts,
        )
