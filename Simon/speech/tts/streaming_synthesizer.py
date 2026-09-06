"""
Streaming Synthesizer for TTS.

Splits input text into sentences/chunks, applies normalization and
pronunciation overrides, checks the cache, and delegates to the underlying
TTS engine. Yields chunks of audio to allow playback to start before
the entire message is synthesized.
"""

from __future__ import annotations

import re
from typing import Iterator

from speech.models.audio_frame import AudioFrame
from speech.tts.base import BaseTTSEngine
from speech.tts.emotional_profile import EmotionalStyle
from speech.tts.pronunciation_dict import PronunciationDictionary
from speech.tts.speech_cache import SpeechCache
from speech.tts.text_normalizer import TextNormalizer


class StreamingSynthesizer:
    """Orchestrates TTS preprocessing, caching, and streaming synthesis."""

    def __init__(
        self,
        engine: BaseTTSEngine,
        cache: SpeechCache,
        normalizer: TextNormalizer,
        pronunciation_dict: PronunciationDictionary,
    ):
        self._engine = engine
        self._cache = cache
        self._normalizer = normalizer
        self._pronunciation_dict = pronunciation_dict
        
        # Simple sentence splitter regex (looks for ., !, ?)
        self._sentence_regex = re.compile(r'(?<=[.!?])\s+')

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into manageable chunks for streaming."""
        # Split by sentence boundaries
        chunks = self._sentence_regex.split(text)
        return [c.strip() for c in chunks if c.strip()]

    def synthesize_stream(self, text: str, style: EmotionalStyle) -> Iterator[AudioFrame]:
        """
        Synthesize text as a stream of AudioFrames.
        
        1. Normalizes text.
        2. Applies pronunciation overrides.
        3. Checks cache.
        4. Chunks text into sentences.
        5. Synthesizes each chunk sequentially.
        """
        if not text.strip():
            return
            
        # 1 & 2. Preprocessing
        norm_text = self._normalizer.normalize(text)
        prep_text = self._pronunciation_dict.apply(norm_text)
        
        # 3. Check cache for the entire prepared string
        cached_frames = self._cache.get(prep_text, self._engine.name, style)
        if cached_frames:
            for frame in cached_frames:
                yield frame
            return
            
        # 4. Chunking (for streaming)
        chunks = self._chunk_text(prep_text)
        all_frames = []
        
        for chunk in chunks:
            # 5. Synthesize
            # Some engines yield one frame, others yield many.
            # We pass them through to the caller immediately.
            for frame in self._engine.synthesize(chunk, style):
                all_frames.append(frame)
                yield frame
                
        # Cache the result for future use if it wasn't aborted
        # (Note: if the generator is closed early, this won't execute)
        if all_frames:
            self._cache.put(prep_text, self._engine.name, style, all_frames)
