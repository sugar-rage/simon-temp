"""Face Registration Tracker — manages async interactions for saving unknown faces.

State machine:

    TRACKING
      ↓ 2 seconds + frontal
    PROMPTING_SAVE
      ├── NO → DECLINED (no re-prompt this encounter)
      └── YES
              ↓
           PROMPTING_NAME
              ├── valid name → COMPLETED (save named)
              ├── no_name / timeout → COMPLETED (save anonymous)
              └── NO → DECLINED
              ↓
           COMPLETED
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from vision.pipeline.perception_fusion import WorldModel

logger = logging.getLogger("simon.core.logic")


class TrackerState(Enum):
    TRACKING = auto()
    PROMPTING_SAVE = auto()
    PROMPTING_NAME = auto()
    COMPLETED = auto()
    DECLINED = auto()


@dataclass
class EncounterState:
    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    state: TrackerState = TrackerState.TRACKING
    embedding: Optional[Any] = None
    prompt_timestamp: Optional[float] = None  # when name question was asked


class FaceRegistrationTracker:
    """Tracks unknown face encounters and triggers registration prompts."""

    def __init__(
        self,
        visibility_threshold_s: float = 2.0,
        grace_period_s: float = 10.0,
        name_timeout_s: float = 15.0,
        face_db_dir: str = "data/faces",
    ) -> None:
        self._threshold = visibility_threshold_s
        self._grace = grace_period_s
        self._name_timeout = name_timeout_s
        self._face_db_dir = face_db_dir
        self._encounters: dict[int, EncounterState] = {}
        self._active_prompt_track_id: Optional[int] = None

    def update_world(self, world: WorldModel) -> Optional[int]:
        """Update tracker with new world state.

        Returns a track_id to prompt for if conditions are met, else None.
        """
        now = time.time()
        active_track_ids = {e.track_id for e in world.entities if e.track_id is not None}

        for entity in world.entities:
            if entity.face_name == "Unknown" and entity.track_id is not None:
                track_id = entity.track_id

                if track_id not in self._encounters:
                    # Check for recently lost encounter to merge
                    matched_tid = self._find_mergeable_encounter(now, active_track_ids)

                    if matched_tid is not None:
                        logger.info(
                            "[FACE_REG] Merged lost track %d into new track %d",
                            matched_tid, track_id,
                        )
                        enc = self._encounters.pop(matched_tid)
                        enc.track_id = track_id
                        self._encounters[track_id] = enc
                        if self._active_prompt_track_id == matched_tid:
                            self._active_prompt_track_id = track_id
                    else:
                        logger.info(
                            "[FACE_REG] Encounter created: track_id=%d",
                            track_id,
                        )
                        self._encounters[track_id] = EncounterState(
                            track_id=track_id,
                            first_seen=now,
                            last_seen=now,
                            embedding=entity.face_embedding,
                        )
                        if entity.face_embedding is not None:
                            logger.info("[FACE_REG] Embedding captured for track_id=%d", track_id)

                enc = self._encounters[track_id]
                enc.last_seen = now
                logger.debug("[FACE_REG] Encounter updated: track_id=%d", track_id)
                if enc.embedding is None and entity.face_embedding is not None:
                    enc.embedding = entity.face_embedding
                    logger.info("[FACE_REG] Embedding captured for track_id=%d", track_id)

                # Only trigger prompt if TRACKING, threshold met, and no other prompt is currently active
                if enc.state == TrackerState.TRACKING:
                    if (now - enc.first_seen) >= self._threshold:
                        if self._active_prompt_track_id is None:
                            enc.state = TrackerState.PROMPTING_SAVE
                            enc.prompt_timestamp = now
                            self._active_prompt_track_id = track_id
                            logger.info(
                                "[FACE_REG] State transition: TRACKING -> PROMPTING_SAVE for track_id=%d",
                                track_id,
                            )
                            logger.info(
                                "[FACE_REG] Prompt triggered: track_id=%d", track_id,
                            )
                            return track_id

        # Clean up stale encounters:
        # - TRACKING, COMPLETED, DECLINED expire after grace_period_s when not seen
        # - PROMPTING_SAVE and PROMPTING_NAME MUST NOT be expired by normal stale cleanup;
        #   they remain alive during the interactive dialogue.
        stale_ids = [
            tid for tid, enc in self._encounters.items()
            if enc.state in (TrackerState.TRACKING, TrackerState.COMPLETED, TrackerState.DECLINED)
            and (now - enc.last_seen) > self._grace
        ]
        for tid in stale_ids:
            logger.info("[FACE_REG] Encounter expired: track_id=%d", tid)
            del self._encounters[tid]
            if self._active_prompt_track_id == tid:
                self._active_prompt_track_id = None

        return None

    def check_name_timeout(self) -> bool:
        """Check if the name-prompt has timed out.

        Returns True if timed out (caller should save anonymously).
        """
        if self._active_prompt_track_id is None:
            return False
        enc = self._encounters.get(self._active_prompt_track_id)
        if enc is None or enc.state != TrackerState.PROMPTING_NAME:
            return False
        if enc.prompt_timestamp is None:
            return False
        timed_out = (time.time() - enc.prompt_timestamp) >= self._name_timeout
        if timed_out:
            logger.info("[FACE_REG] Timeout reached for track_id=%s", self._active_prompt_track_id)
        return timed_out

    def get_active_prompt_state(self) -> Optional[TrackerState]:
        if self._active_prompt_track_id and self._active_prompt_track_id in self._encounters:
            return self._encounters[self._active_prompt_track_id].state
        return None

    def advance_to_name_prompt(self) -> None:
        if self._active_prompt_track_id and self._active_prompt_track_id in self._encounters:
            enc = self._encounters[self._active_prompt_track_id]
            enc.state = TrackerState.PROMPTING_NAME
            enc.prompt_timestamp = time.time()
            logger.info(
                "[FACE_REG] State transition: PROMPTING_SAVE -> PROMPTING_NAME for track_id=%d",
                self._active_prompt_track_id,
            )

    def mark_completed(self) -> None:
        if self._active_prompt_track_id and self._active_prompt_track_id in self._encounters:
            self._encounters[self._active_prompt_track_id].state = TrackerState.COMPLETED
            logger.info(
                "[FACE_REG] State transition: -> COMPLETED for track_id=%d",
                self._active_prompt_track_id,
            )
        self._active_prompt_track_id = None

    def mark_declined(self) -> None:
        """User said NO — mark declined so we never re-prompt this encounter."""
        if self._active_prompt_track_id and self._active_prompt_track_id in self._encounters:
            self._encounters[self._active_prompt_track_id].state = TrackerState.DECLINED
            logger.info(
                "[FACE_REG] State transition: -> DECLINED for track_id=%d",
                self._active_prompt_track_id,
            )
        self._active_prompt_track_id = None

    def get_active_embedding(self) -> Optional[Any]:
        if self._active_prompt_track_id and self._active_prompt_track_id in self._encounters:
            return self._encounters[self._active_prompt_track_id].embedding
        return None

    def get_active_track_id(self) -> Optional[int]:
        return self._active_prompt_track_id

    def clear_active_prompt(self) -> None:
        self._active_prompt_track_id = None

    def get_next_anonymous_name(self) -> str:
        """Generate the next anonymous identifier like 'Person 1', 'Person 2'."""
        existing = 0
        if os.path.isdir(self._face_db_dir):
            for name in os.listdir(self._face_db_dir):
                if name.startswith("Person "):
                    try:
                        num = int(name.split(" ", 1)[1])
                        existing = max(existing, num)
                    except (ValueError, IndexError):
                        pass
        return f"Person {existing + 1}"

    # ── Private helpers ──────────────────────────────────────────────

    def _find_mergeable_encounter(self, now: float, active_track_ids: set[int]) -> Optional[int]:
        """Find a recently lost encounter that can be merged."""
        # 1. Active prompt dialogue has highest priority: if an encounter is in PROMPTING_SAVE or PROMPTING_NAME,
        # merge any new unknown track into it so the active dialogue and original embedding survive face reacquisition.
        if self._active_prompt_track_id is not None and self._active_prompt_track_id in self._encounters:
            active_enc = self._encounters[self._active_prompt_track_id]
            if active_enc.state in (TrackerState.PROMPTING_SAVE, TrackerState.PROMPTING_NAME):
                if self._active_prompt_track_id not in active_track_ids:
                    return self._active_prompt_track_id

        # 2. General recently lost tracking encounters (within 2s)
        for tid, enc in self._encounters.items():
            if tid not in active_track_ids and enc.state == TrackerState.TRACKING:
                if 0.0 < (now - enc.last_seen) < 2.0:
                    return tid
        return None
