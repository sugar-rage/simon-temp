"""
User Profile — persistent user preferences for the SIMON speech subsystem.

Stores per-user settings that accumulate over time:
- Preferred TTS voice and speaking rate
- Preferred volume level
- Accent hint for Whisper (e.g. ``"Indian English"``)
- Custom corrections history
- Enrollment status (speaker verification)

Persisted as a JSON file so it survives restarts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Persistent user preference profile.

    Attributes:
        user_id:            Unique user identifier.
        display_name:       Human-readable name.
        accent_hint:        Accent descriptor for Whisper prompt (e.g. "Indian English").
        preferred_speed:    TTS speaking rate multiplier (1.0 = normal).
        preferred_volume:   Volume 0–100.
        tts_voice_id:       Preferred TTS voice identifier.
        corrections:        History of user corrections ``{wrong: right}``.
        enrolled:           Whether speaker verification enrollment is complete.
        created_at:         Unix timestamp of profile creation.
        updated_at:         Unix timestamp of last modification.
    """

    user_id: str = "default"
    display_name: str = "User"
    accent_hint: str = ""
    preferred_speed: float = 1.0
    preferred_volume: int = 75
    tts_voice_id: str = ""
    corrections: Dict[str, str] = field(default_factory=dict)
    enrolled: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_correction(self, wrong: str, right: str) -> None:
        """Record a user correction (e.g. user says 'I meant X not Y')."""
        self.corrections[wrong.lower()] = right
        self.updated_at = time.time()

    def get_correction(self, text: str) -> Optional[str]:
        """Look up a known correction."""
        return self.corrections.get(text.lower())

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class UserProfileManager:
    """Manages loading, saving, and switching user profiles.

    Args:
        profile_dir: Directory for profile JSON files.
    """

    def __init__(self, profile_dir: str = "data/profiles"):
        self._dir = Path(profile_dir)
        self._profiles: Dict[str, UserProfile] = {}
        self._active_id: str = "default"

    @property
    def active_profile(self) -> UserProfile:
        """Get the currently active user profile."""
        if self._active_id not in self._profiles:
            self._profiles[self._active_id] = UserProfile(user_id=self._active_id)
        return self._profiles[self._active_id]

    def set_active(self, user_id: str) -> UserProfile:
        """Switch active user. Loads from disk if not in memory."""
        self._active_id = user_id
        if user_id not in self._profiles:
            profile = self._load(user_id)
            if profile:
                self._profiles[user_id] = profile
            else:
                self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def save(self, user_id: Optional[str] = None) -> None:
        """Persist a profile to disk."""
        uid = user_id or self._active_id
        profile = self._profiles.get(uid)
        if not profile:
            return

        profile.updated_at = time.time()
        path = self._dir / f"{uid}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(profile.to_dict(), indent=2),
                encoding="utf-8",
            )
            logger.debug(f"Saved profile: {uid}")
        except OSError as e:
            logger.warning(f"Failed to save profile {uid}: {e}")

    def save_all(self) -> None:
        """Persist all loaded profiles."""
        for uid in self._profiles:
            self.save(uid)

    def list_profiles(self) -> List[str]:
        """List all known profile IDs (from memory and disk)."""
        ids = set(self._profiles.keys())
        if self._dir.exists():
            for f in self._dir.glob("*.json"):
                ids.add(f.stem)
        return sorted(ids)

    def _load(self, user_id: str) -> Optional[UserProfile]:
        """Load a profile from disk."""
        path = self._dir / f"{user_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.debug(f"Loaded profile: {user_id}")
            return UserProfile.from_dict(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load profile {user_id}: {e}")
            return None
