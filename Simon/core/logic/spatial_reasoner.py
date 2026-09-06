"""Spatial Reasoner — computes spatial relationships between entities.

Generates human-friendly spatial descriptions like:
- "car to your left, near"
- "person in the center, approaching"
- "2 cars ahead, 1 person to your right"
"""

from __future__ import annotations

import logging
from typing import Optional

from vision.pipeline.perception_fusion import Entity, WorldModel

logger = logging.getLogger("simon.core.logic")


class SpatialReasoner:
    """Computes spatial descriptions for world entities.

    Converts raw position/distance data into natural language descriptions
    suitable for speech announcements.
    """

    def describe_entity(self, entity: Entity) -> str:
        """Generate a spatial description for a single entity.

        Examples::

            "car to your left, near"
            "Alice ahead, medium distance"
            "stop sign to your right"
        """
        parts: list[str] = []

        # Identity / class
        if entity.face_name and entity.face_name != "Unknown":
            parts.append(entity.face_name)
        else:
            parts.append(entity.cls)

        # Position
        pos = entity.position
        if pos == "left":
            parts.append("to your left")
        elif pos in ("slightly_left", "slightly left"):
            parts.append("slightly to your left")
        elif pos == "right":
            parts.append("to your right")
        elif pos in ("slightly_right", "slightly right"):
            parts.append("slightly to your right")
        elif pos in ("center", "ahead"):
            parts.append("ahead")

        # Distance
        if entity.distance == "near":
            parts.append("nearby")
        elif entity.distance == "far":
            parts.append("in the distance")

        return ", ".join(parts)

    def describe_scene(self, world: WorldModel) -> str:
        """Generate a brief scene summary.

        Examples::

            "Street scene: 3 cars, 2 people, 1 traffic light"
            "Indoor: chair, couch, TV"
        """
        if not world.entities and not world.scene:
            return "Nothing detected"

        parts: list[str] = []

        # Scene type
        if world.scene and world.scene.scene_type != "unknown":
            parts.append(f"{world.scene.scene_type.title()} scene")

        # Object summary — group by class
        class_counts: dict[str, int] = {}
        for entity in world.entities:
            cls = entity.face_name if entity.face_name and entity.face_name != "Unknown" else entity.cls
            class_counts[cls] = class_counts.get(cls, 0) + 1

        if class_counts:
            summaries = []
            for cls, count in sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            ):
                if count > 1:
                    summaries.append(f"{count} {cls}s")
                else:
                    summaries.append(cls)
            parts.append(", ".join(summaries[:5]))  # top 5

        # OCR text
        if world.ocr_texts:
            texts = [t.text for t in world.ocr_texts[:3]]
            parts.append(f"text: {', '.join(texts)}")

        return ". ".join(parts) if parts else "Nothing detected"

    def get_hazard_description(self, entity: Entity) -> str:
        """Generate an urgent hazard description for TTS.

        Examples::

            "Warning! Car approaching from the left!"
            "Caution: bicycle ahead"
        """
        prefix = ""
        if entity.hazard_level is not None:
            from core.models.enums import HazardLevel

            if entity.hazard_level <= HazardLevel.CRITICAL:
                prefix = "Warning! "
            elif entity.hazard_level <= HazardLevel.WARNING:
                prefix = "Caution: "

        desc = self.describe_entity(entity)
        return f"{prefix}{desc}"
