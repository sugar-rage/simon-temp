"""Language Generator — converts structured decisions into speech-ready text.

Provides templates and natural language generation for all announcement
types: detection, navigation, safety, status.
"""

from __future__ import annotations

import random
from typing import Optional

from vision.pipeline.perception_fusion import Entity, WorldModel


class LanguageGenerator:
    """Generates natural language text for TTS announcements.

    Templates use variation to avoid repetitive announcements.
    """

    # ── Detection Announcements ──────────────────────────────────────

    _DETECTION_TEMPLATES = [
        "{cls} {position}",
        "{cls} detected {position}",
        "I see a {cls} {position}",
    ]

    _FACE_TEMPLATES = [
        "{name} is {position}",
        "I see {name} {position}",
        "{name} detected {position}",
    ]

    _HAZARD_TEMPLATES = {
        1: [  # CRITICAL
            "Warning! {cls} very close, {position}!",
            "Danger! {cls} {position}, move aside!",
            "Watch out! {cls} approaching {position}!",
        ],
        2: [  # WARNING
            "Caution, {cls} {position}",
            "Be careful, {cls} {position}",
            "{cls} nearby {position}",
        ],
        3: [  # CAUTION
            "{cls} ahead {position}",
            "{cls} detected {position}",
        ],
    }

    def announce_entity(self, entity: Entity) -> Optional[str]:
        """Generate an announcement for a detected entity.

        Returns None if the entity doesn't warrant an announcement.
        """
        position = self._position_text(entity.position)

        # Hazard announcement
        if entity.hazard_level is not None and entity.hazard_level <= 3:
            templates = self._HAZARD_TEMPLATES.get(
                entity.hazard_level, self._HAZARD_TEMPLATES[3]
            )
            return random.choice(templates).format(
                cls=entity.cls, position=position
            )

        # Known face announcement
        if entity.face_name:
            if entity.face_name == "Unknown":
                return f"Unknown person {position}."
            return self.known_person_arrival(
                name=entity.face_name,
                position=entity.position,
                distance=entity.distance,
            )

        # Generic detection announcement
        return random.choice(self._DETECTION_TEMPLATES).format(
            cls=entity.cls, position=position
        )

    def known_person_arrival(
        self,
        name: str,
        position: Optional[str] = None,
        distance: Optional[str] = None,
    ) -> str:
        """Format an arrival announcement for a recognized person with spatial information.

        Examples
        --------
        - "Ganesh is here, on your left."
        - "Ganesh is here, on your right."
        - "Ganesh is here, directly ahead."
        - "Ganesh is here, nearby on your left."
        - "Ganesh is here, far ahead."
        - "Ganesh is here."
        """
        # Direction phrase
        dir_text = None
        pos_lower = position.lower().strip() if position else None
        if pos_lower:
            if pos_lower == "left":
                dir_text = "on your left"
            elif pos_lower in ("slightly_left", "slightly left"):
                dir_text = "slightly on your left"
            elif pos_lower in ("center", "ahead"):
                dir_text = "directly ahead"
            elif pos_lower in ("slightly_right", "slightly right"):
                dir_text = "slightly on your right"
            elif pos_lower == "right":
                dir_text = "on your right"

        # Distance phrase
        dist_text = None
        dist_lower = distance.lower().strip() if distance else None
        if dist_lower:
            if dist_lower == "near":
                dist_text = "nearby"
            elif dist_lower == "far":
                if pos_lower in ("center", "ahead", None):
                    dist_text = "far ahead"
                else:
                    dist_text = "in the distance"

        # Combine spatial components
        spatial_part = None
        if dist_text and dir_text:
            if dist_text == "nearby":
                if "on your" in dir_text:
                    spatial_part = f"nearby {dir_text}"
                elif dir_text == "directly ahead":
                    spatial_part = "nearby directly ahead"
                else:
                    spatial_part = f"nearby {dir_text}"
            elif dist_text == "far ahead" and dir_text == "directly ahead":
                spatial_part = "far ahead"
            else:
                spatial_part = f"{dir_text}, {dist_text}"
        elif dir_text:
            spatial_part = dir_text
        elif dist_text:
            spatial_part = dist_text

        if spatial_part:
            return f"{name} is here, {spatial_part}."
        return f"{name} is here."

    # ── Navigation Announcements ─────────────────────────────────────

    def navigation_instruction(
        self,
        instruction: str,
        distance_m: float,
    ) -> str:
        """Format a navigation instruction for TTS."""
        if distance_m < 10:
            return instruction
        elif distance_m < 100:
            return f"In {int(distance_m)} meters, {instruction}"
        else:
            return f"In {int(distance_m)} meters, {instruction}"

    def arrival_announcement(self, destination: str) -> str:
        """Generate arrival announcement."""
        templates = [
            f"You have arrived at {destination}",
            f"Destination reached: {destination}",
            f"We're here. {destination}",
        ]
        return random.choice(templates)

    def off_route_announcement(self, distance_m: float) -> str:
        """Generate off-route warning."""
        return f"You appear to be off route by {int(distance_m)} meters. Recalculating."

    # ── Status Announcements ─────────────────────────────────────────

    def scene_summary(self, world: WorldModel) -> str:
        """Generate a scene summary for on-demand 'what do you see?' queries."""
        if not world.entities:
            return "I don't see anything right now"

        class_counts: dict[str, int] = {}
        for entity in world.entities:
            label = (
                entity.face_name
                if entity.face_name
                else entity.cls
            )
            class_counts[label] = class_counts.get(label, 0) + 1

        parts = []
        for cls, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        )[:6]:
            if count > 1:
                parts.append(f"{count} {cls}s")
            else:
                parts.append(f"a {cls}")

        scene_type = ""
        if world.scene and world.scene.scene_type != "unknown":
            scene_type = f"It looks like a {world.scene.scene_type} scene. "

        return f"{scene_type}I can see {', '.join(parts)}"

    def capability_announcement(self, available: list[str], degraded: list[str]) -> str:
        """Announce system capabilities on startup."""
        parts = []
        if available:
            parts.append(f"{len(available)} features ready")
        if degraded:
            names = ", ".join(d.replace("capability.", "") for d in degraded)
            parts.append(f"{names} running in limited mode")
        return "System started. " + ". ".join(parts) if parts else "System started"

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _position_text(position: Optional[str]) -> str:
        """Convert position value to speech-friendly text."""
        if position == "left":
            return "to your left"
        elif position == "right":
            return "to your right"
        elif position == "center":
            return "ahead"
        return ""
