"""
Speech Cache for TTS.

Caches synthesized audio for frequently repeated phrases (e.g., "Yes", "Okay",
"Navigation stopped") to eliminate synthesis latency.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from speech.models.audio_frame import AudioFrame
from speech.tts.emotional_profile import EmotionalStyle


class SpeechCache:
    """In-memory and disk cache for synthesized speech."""

    def __init__(self, cache_dir: Optional[str] = None, max_entries: int = 100):
        self._cache: Dict[str, List[AudioFrame]] = {}
        self._max_entries = max_entries
        self._cache_dir = Path(cache_dir) if cache_dir else None
        
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _generate_key(self, text: str, engine_name: str, style: EmotionalStyle) -> str:
        """Generate a unique cache key for the text and style."""
        # Include text, engine, and style attributes in the hash
        raw_key = f"{text}|{engine_name}|{style.speed_factor}|{style.pitch_shift_pct}|{style.volume_pct}"
        return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

    def get(self, text: str, engine_name: str, style: EmotionalStyle) -> Optional[List[AudioFrame]]:
        """Retrieve cached audio frames if available."""
        key = self._generate_key(text, engine_name, style)
        return self._cache.get(key)

    def put(self, text: str, engine_name: str, style: EmotionalStyle, frames: List[AudioFrame]) -> None:
        """Store synthesized frames in the cache."""
        # Don't cache extremely long texts
        if len(text) > 200:
            return
            
        key = self._generate_key(text, engine_name, style)
        
        # Simple LRU-ish eviction: if over limit, remove arbitrary item (first key)
        if len(self._cache) >= self._max_entries and key not in self._cache:
            first_key = next(iter(self._cache.keys()))
            del self._cache[first_key]
            
        self._cache[key] = frames
        self._save_to_disk()

    def _load_from_disk(self) -> None:
        """Load cache entries from disk."""
        if not self._cache_dir:
            return
            
        cache_file = self._cache_dir / "tts_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception:
                # If cache is corrupted, just start fresh
                self._cache = {}

    def _save_to_disk(self) -> None:
        """Persist cache entries to disk."""
        if not self._cache_dir:
            return
            
        cache_file = self._cache_dir / "tts_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        if self._cache_dir:
            cache_file = self._cache_dir / "tts_cache.pkl"
            if cache_file.exists():
                cache_file.unlink()
