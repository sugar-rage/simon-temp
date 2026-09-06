"""
pyttsx3 TTS Engine — The reliable offline fallback.

Uses the OS-native TTS voice (SAPI5 on Windows, NSSpeech on Mac, eSpeak on Linux).
Extremely fast but robotic. Does not support streaming chunks, so it
yields the entire audio sentence at once.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterator

import pyttsx3
import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.tts.base import BaseTTSEngine
from speech.tts.emotional_profile import EmotionalStyle


logger = logging.getLogger(__name__)


class Pyttsx3Engine(BaseTTSEngine):
    """Fallback TTS engine using pyttsx3 (OS native voices)."""

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_id: str = ""):
        self._default_rate = rate
        self._default_volume = volume
        self._voice_id = voice_id
        
        self._engine = None
        self._lock = threading.Lock()
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "pyttsx3"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            return
            
        with self._lock:
            try:
                test_engine = pyttsx3.init()
                test_engine.stop()
                del test_engine
                self._is_loaded = True
                logger.info(f"Loaded {self.name} TTS engine successfully.")
            except Exception as e:
                logger.error(f"Failed to load {self.name} TTS engine: {e}")
                raise

    def unload(self) -> None:
        with self._lock:
            self._is_loaded = False
            logger.info(f"Unloaded {self.name} TTS engine.")

    def synthesize(self, text: str, style: EmotionalStyle) -> Iterator[AudioFrame]:
        """Synthesize text to audio. Yields one chunk containing the whole sentence."""
        if not text or not text.strip():
            return

        with self._lock:
            # Apply emotional style adjustments
            target_rate = int(self._default_rate * style.speed_factor)
            target_volume = min(1.0, max(0.0, self._default_volume * (style.volume_pct / 100.0)))
            
            engine = None
            try:
                import tempfile
                import wave
                import os
                
                # To prevent SAPI5 COM stream / event-loop deadlock on Windows when
                # save_to_file() is called repeatedly across utterances, create and
                # clean up a dedicated pyttsx3 engine instance per synthesis.
                engine = pyttsx3.init()
                engine.setProperty("rate", target_rate)
                engine.setProperty("volume", target_volume)
                
                if self._voice_id:
                    engine.setProperty("voice", self._voice_id)
                
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                
                try:
                    engine.save_to_file(text, path)
                    engine.runAndWait()
                    
                    if os.path.exists(path) and os.path.getsize(path) > 44:
                        with wave.open(path, "rb") as wf:
                            sample_rate = wf.getframerate()
                            channels = wf.getnchannels()
                            sampwidth = wf.getsampwidth()
                            
                            raw_data = wf.readframes(wf.getnframes())
                            
                            if sampwidth == 2:
                                dtype = np.int16
                            else:
                                dtype = np.float32  # Approximation
                                
                            data = np.frombuffer(raw_data, dtype=dtype)
                            
                            yield AudioFrame(
                                data=data,
                                sample_rate=sample_rate,
                                channels=channels,
                                dtype="int16" if sampwidth == 2 else "float32"
                            )
                finally:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                            
            except Exception as e:
                logger.error(f"[SPEECH] Pyttsx3 synthesis failed: {e}", exc_info=True)
                
            finally:
                if engine:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    del engine
