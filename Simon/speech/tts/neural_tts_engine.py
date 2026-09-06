"""
Neural TTS Engine using Coqui XTTS-v2.

Provides near-human quality speech with emotional styling (via reference audio
or prompt engineering). Supports streaming synthesis to reduce latency.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.tts.base import BaseTTSEngine
from speech.tts.emotional_profile import EmotionalStyle


logger = logging.getLogger(__name__)


class NeuralTTSEngine(BaseTTSEngine):
    """High-quality neural TTS engine (Coqui XTTS-v2 wrapper)."""

    def __init__(
        self, 
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        use_gpu: bool = True,
        speaker_wav: Optional[str] = None,
        language: str = "en"
    ):
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._speaker_wav = speaker_wav
        self._language = language
        
        self._model = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "xtts_v2"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            return
            
        try:
            # Import TTS locally so it's not a hard dependency if user only uses fallback
            from TTS.api import TTS
            import torch
            
            # Auto-detect GPU if requested
            device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {self.name} on {device}...")
            
            self._model = TTS(self._model_name).to(device)
            self._is_loaded = True
            logger.info(f"Loaded {self.name} successfully.")
            
        except ImportError:
            logger.error("TTS package is not installed. Run `pip install TTS` to use XTTS-v2.")
            raise
        except Exception as e:
            logger.error(f"Failed to load Neural TTS: {e}")
            raise

    def unload(self) -> None:
        if self._model:
            del self._model
            self._model = None
            
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        self._is_loaded = False
        logger.info(f"Unloaded {self.name}.")

    def synthesize(self, text: str, style: EmotionalStyle) -> Iterator[AudioFrame]:
        """Synthesize text into audio chunks for streaming playback."""
        if not self._is_loaded:
            self.load()
            
        if not self._model:
            raise RuntimeError("Model failed to load.")
            
        # Neural TTS streaming approach:
        # XTTS-v2 has a synthesis generator method `synthesize_stream` or we can chunk sentences.
        # For simplicity and robustness, we can assume text is already chunked at sentence level
        # by the streaming_synthesizer.py, so we just synthesize this chunk.
        
        try:
            # For this wrapper, we just call the basic TTS output.
            # True streaming XTTS requires accessing the underlying model's inference generator.
            # We will use the standard API and yield the full sentence as one chunk, relying
            # on `streaming_synthesizer.py` to chunk the input text.
            
            # Apply basic style modifications via speed factor if supported, or rely on speaker_wav
            # for emotional cloning.
            
            wav = self._model.tts(
                text=text, 
                speaker_wav=self._speaker_wav, 
                language=self._language,
                speed=style.speed_factor
            )
            
            # TTS API returns a list or numpy array of floats usually at 24000Hz for XTTS
            # Check model properties for sample rate
            sample_rate = self._model.synthesizer.output_sample_rate
            
            audio_data = np.array(wav, dtype=np.float32)
            
            yield AudioFrame(
                data=audio_data,
                sample_rate=sample_rate,
                channels=1,
                dtype="float32"
            )
            
        except Exception as e:
            logger.error(f"Neural TTS synthesis failed: {e}")
            # Could raise or return silence
