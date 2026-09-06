"""Priority Engine — assigns announcement priorities to world entities.

Determines which entities should be announced based on novelty, distance,
hazard level, and context.  Replaces the legacy ``logic/merger.py`` priority
assignment with a configurable, testable engine.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.models.enums import Priority, HazardLevel
from vision.pipeline.perception_fusion import Entity, WorldModel

logger = logging.getLogger("simon.core.logic")


class PriorityEngine:
    """Assigns announcement priority scores to world entities.

    Scoring factors (lower = higher priority):
    1. Hazard level (critical > warning > caution > info)
    2. Distance (near > medium > far)
    3. Novelty (new entities score higher than persistent ones)
    4. Class importance (vehicles, people, signs > ambient objects)
    5. Face recognition (known faces boost priority)

    Parameters
    ----------
    high_priority_classes : set[str]
        Object classes that get higher priority.
    """

    def __init__(
        self,
        high_priority_classes: Optional[set[str]] = None,
    ) -> None:
        self._high_priority_classes = high_priority_classes or {
            "car", "truck", "bus", "motorcycle", "bicycle",
            "person", "traffic light", "stop sign", "fire hydrant",
        }

    def score(self, entity: Entity) -> int:
        """Compute announcement priority for an entity.

        Returns a Priority value (lower = more urgent).
        """
        # Start with base priority
        base = Priority.INFORMATIONAL

        # Factor 1: Hazard level
        if entity.hazard_level is not None:
            if entity.hazard_level <= HazardLevel.CRITICAL:
                return Priority.EMERGENCY
            elif entity.hazard_level <= HazardLevel.WARNING:
                return Priority.SAFETY_CRITICAL
            elif entity.hazard_level <= HazardLevel.CAUTION:
                base = Priority.OBSTACLE

        # Factor 2: Known face (named or anonymous Person N, not Unknown)
        if entity.face_name and entity.face_name != "Unknown":
            return Priority.FACE

        # Factor 3: Distance
        if entity.distance == "near":
            base = min(base, Priority.OBSTACLE)
        elif entity.distance == "medium":
            base = min(base, Priority.NAVIGATION)

        # Factor 4: Class importance
        if entity.cls in self._high_priority_classes:
            base = min(base, Priority.NAVIGATION)

        return base

    def rank_entities(self, world: WorldModel) -> list[Entity]:
        """Sort world entities by announcement priority (most urgent first).

        Returns entities that should be announced, filtered and ranked.
        """
        if not world.entities:
            return []

        scored: list[tuple[int, Entity]] = [
            (self.score(entity), entity) for entity in world.entities
        ]

        # Sort by score (ascending = most urgent first)
        scored.sort(key=lambda x: x[0])

        return [entity for _, entity in scored]
