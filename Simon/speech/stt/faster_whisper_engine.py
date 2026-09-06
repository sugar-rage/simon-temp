"""
faster-whisper STT engine — production primary ASR engine.

Wraps ``faster_whisper.WhisperModel`` (CTranslate2 backend) to provide
4× faster inference than OpenAI Whisper with identical accuracy.  Uses
the Large-v3 Turbo model by default for the best accuracy/speed tradeoff.

This module:
- Auto-detects GPU/CPU and selects compute type accordingly
- Accepts ``DecodingParams`` from the adaptive decoder
- Returns rich ``Transcript`` objects with per-word timestamps
- Handles model loading failures gracefully
- Reports detailed latency and confidence metrics
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from speech.errors.exceptions import (
    GPUUnavailableError,
    STTModelLoadError,
    STTTranscriptionError,
)
from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.stt.base import BaseSTTEngine
from speech.stt.transcript import Transcript, WordInfo

logger = get_logger("stt.faster_whisper")
metrics = get_collector()


class FasterWhisperEngine(BaseSTTEngine):
    """faster-whisper (CTranslate2) STT engine.

    Args:
        model_size:   Model size string (e.g. "large-v3", "medium", "small").
        compute_type: CTranslate2 compute type ("float16", "int8", "float32").
        device:       "auto", "cuda", or "cpu".
        language:     Language code or None for auto-detection.
    """

    def __init__(
        self,
        model_size: str = "medium",
        compute_type: str = "float16",
        device: str = "auto",
        language: str = "en",
    ):
        self._model_size = model_size
        self._compute_type = compute_type
        self._device_preference = device
        self._language = language

        self._model = None
        self._is_loaded = False
        self._actual_device = "cpu"
        self._actual_compute_type = compute_type

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def supports_word_timestamps(self) -> bool:
        return True

    def _resolve_device(self) -> tuple[str, str]:
        """Determine device and compute type based on availability.

        Returns:
            Tuple of (device, compute_type).
        """
        if self._device_preference == "cpu":
            return "cpu", "float32"

        # Try CUDA
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                logger.info(f"CUDA available, VRAM: {vram_mb}MB")

                if vram_mb < 2000:
                    return "cuda", "int8"
                elif vram_mb < 6000:
                    return "cuda", "int8"
                else:
                    return "cuda", self._compute_type
        except ImportError:
            pass

        # Fallback to CPU
        logger.info("CUDA not available, using CPU with float32")
        return "cpu", "float32"

    def load(self) -> None:
        """Load the faster-whisper model."""
        if self._is_loaded:
            return

        self._actual_device, self._actual_compute_type = self._resolve_device()

        try:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading faster-whisper model",
                extra={
                    "model": self._model_size,
                    "device": self._actual_device,
                    "compute_type": self._actual_compute_type,
                },
            )

            t0 = time.monotonic()
            self._model = WhisperModel(
                self._model_size,
                device=self._actual_device,
                compute_type=self._actual_compute_type,
            )
            load_time = time.monotonic() - t0

            self._is_loaded = True
            metrics.histogram("stt.model_load_time_s", load_time)
            logger.info(
                f"faster-whisper loaded in {load_time:.1f}s",
                extra={"device": self._actual_device, "compute": self._actual_compute_type},
            )

        except ImportError:
            raise STTModelLoadError(
                "faster-whisper", ImportError("pip install faster-whisper")
            )
        except Exception as e:
            # If CUDA fails, retry on CPU
            if self._actual_device == "cuda":
                logger.warning(f"CUDA load failed: {e}, falling back to CPU")
                self._actual_device = "cpu"
                self._actual_compute_type = "float32"
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(
                        self._model_size,
                        device="cpu",
                        compute_type="float32",
                    )
                    self._is_loaded = True
                    logger.info("faster-whisper loaded on CPU fallback")
                    return
                except Exception as e2:
                    raise STTModelLoadError(self._model_size, e2) from e2
            raise STTModelLoadError(self._model_size, e) from e

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        self._is_loaded = False
        logger.info("faster-whisper model unloaded")

        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(
        self,
        audio: AudioFrame,
        params: Optional[DecodingParams] = None,
    ) -> Transcript:
        """Transcribe an audio segment using faster-whisper.

        Args:
            audio:  AudioFrame containing speech (any sample rate/dtype).
            params: Decoding parameters.  If None, defaults are used.

        Returns:
            A ``Transcript`` with text, confidence, word timestamps, etc.

        Raises:
            STTTranscriptionError: If transcription fails.
        """
        if not self._is_loaded:
            self.load()

        # Ensure float32 at 16kHz for Whisper
        audio_f32 = audio.to_float32()
        if audio_f32.sample_rate != 16000:
            audio_f32 = audio_f32.resample(16000)

        params = params or DecodingParams()

        t0 = time.monotonic()

        try:
            segments_iter, info = self._model.transcribe(
                audio_f32.data,
                beam_size=params.beam_size,
                best_of=params.best_of,
                patience=params.patience,
                length_penalty=params.length_penalty,
                temperature=params.temperature,
                compression_ratio_threshold=params.compression_ratio_threshold,
                log_prob_threshold=params.log_prob_threshold,
                no_speech_threshold=params.no_speech_threshold,
                initial_prompt=params.initial_prompt,
                language=params.language,
                word_timestamps=True,
                vad_filter=True,
            )

            # Collect all segments
            full_text_parts: list[str] = []
            all_words: list[WordInfo] = []
            total_log_prob = 0.0
            segment_count = 0
            max_compression_ratio = 0.0
            max_no_speech_prob = 0.0

            for segment in segments_iter:
                full_text_parts.append(segment.text.strip())
                segment_count += 1
                total_log_prob += segment.avg_logprob
                max_compression_ratio = max(max_compression_ratio, segment.compression_ratio)
                max_no_speech_prob = max(max_no_speech_prob, segment.no_speech_prob)

                if segment.words:
                    for w in segment.words:
                        all_words.append(WordInfo(
                            word=w.word.strip(),
                            start=w.start,
                            end=w.end,
                            confidence=w.probability if hasattr(w, "probability") else 0.0,
                        ))

            full_text = " ".join(full_text_parts).strip()
            avg_log_prob = total_log_prob / max(segment_count, 1)
            confidence = Transcript.logprob_to_confidence(avg_log_prob)

            latency_s = time.monotonic() - t0

            transcript = Transcript(
                text=full_text.lower(),
                raw_text=full_text,
                confidence=confidence,
                language=info.language if info.language else params.language or "en",
                language_confidence=info.language_probability if hasattr(info, "language_probability") else 0.0,
                words=all_words,
                duration_s=audio.duration_s,
                latency_s=latency_s,
                engine_name=self.name,
                temperature=params.temperature,
                avg_log_prob=avg_log_prob,
                compression_ratio=max_compression_ratio,
                no_speech_prob=max_no_speech_prob,
            )

            # Record metrics
            metrics.histogram("stt.latency_ms", latency_s * 1000)
            metrics.gauge("stt.confidence", confidence)
            metrics.gauge("stt.rtf", transcript.real_time_factor)
            metrics.label("stt.engine_active", self.name)

            logger.info(
                f"Transcribed: '{full_text[:60]}{'...' if len(full_text) > 60 else ''}'",
                extra={
                    "confidence": f"{confidence:.2f}",
                    "latency_ms": f"{latency_s * 1000:.0f}",
                    "rtf": f"{transcript.real_time_factor:.2f}",
                    "words": len(all_words),
                },
            )

            return transcript

        except Exception as e:
            latency_s = time.monotonic() - t0
            logger.error(f"Transcription failed after {latency_s:.1f}s: {e}")
            metrics.counter("stt.transcription_errors")
            raise STTTranscriptionError(self.name, e) from e
