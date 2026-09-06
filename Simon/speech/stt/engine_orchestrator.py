"""
Multi-Engine STT Orchestrator — the central intelligence for speech recognition.

Routes audio through a primary STT engine, evaluates confidence, triggers
intelligent redecoding when confidence is low, and falls back to secondary
engines if the primary cannot produce an acceptable transcript.

This is the single point of failure mitigation for a safety-critical system:
a misrecognition of "stop" as "shop" during navigation could endanger the user.
The orchestrator provides redundancy through a fallback chain and consensus.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.monitoring.metrics import get_collector
from speech.stt.base import BaseSTTEngine
from speech.stt.engine_registry import EngineRegistry
from speech.stt.intelligent_redecoder import IntelligentRedecoder
from speech.stt.transcript import Transcript

logger = logging.getLogger(__name__)
metrics = get_collector()


class EngineOrchestrator:
    """Multi-engine STT orchestrator with confidence-gated fallback.

    Pipeline:
        1. Primary engine transcribes at temperature=0.0
        2. If confidence >= high_threshold → return immediately
        3. If confidence < high_threshold → IntelligentRedecoder retries
        4. If still < low_threshold → try fallback engines in priority order
        5. Select best transcript across all engines

    Args:
        registry: EngineRegistry managing available engines.
        redecoder: IntelligentRedecoder for multi-temperature retry.
        high_confidence_threshold: Skip redecoding above this.
        low_confidence_threshold: Trigger fallback below this.
    """

    def __init__(
        self,
        registry: EngineRegistry,
        redecoder: Optional[IntelligentRedecoder] = None,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.50,
    ):
        self._registry = registry
        self._redecoder = redecoder or IntelligentRedecoder(
            high_confidence_threshold=high_confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
        )
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold

    @property
    def registry(self) -> EngineRegistry:
        return self._registry

    def transcribe(
        self,
        audio: AudioFrame,
        params: Optional[DecodingParams] = None,
    ) -> Transcript:
        """Transcribe audio using the orchestrated multi-engine pipeline.

        Args:
            audio: Audio segment to transcribe.
            params: Decoding parameters (scene-adapted). Defaults used if None.

        Returns:
            The best Transcript available from all engines.
        """
        params = params or DecodingParams()
        start_time = time.monotonic()

        # 1. Get primary engine
        primary = self._registry.get_primary()
        if primary is None:
            logger.error("No STT engines available in registry")
            metrics.counter("stt.no_engine_available")
            return Transcript()

        # 2. Ensure primary is loaded
        if not primary.is_loaded:
            try:
                primary.load()
            except Exception as e:
                logger.error(f"Failed to load primary engine '{primary.name}': {e}")
                return self._try_fallbacks(audio, params)

        # 3. Run primary engine through the redecoder
        primary_health = self._registry.get_health(primary.name)

        try:
            result = self._redecoder.decode_with_retry(primary, audio, params)

            # Record success
            if primary_health:
                primary_health.record_success()

        except Exception as e:
            logger.error(f"Primary engine '{primary.name}' failed: {e}")
            if primary_health:
                primary_health.record_failure()
            metrics.counter("stt.primary_engine_error")
            return self._try_fallbacks(audio, params)

        best_transcript = result.best

        # 4. If redecoder says we need fallback, try other engines
        if result.needs_fallback:
            logger.info(
                f"Primary engine confidence too low ({best_transcript.confidence:.3f}), "
                f"trying fallback engines"
            )
            metrics.counter("stt.fallback_triggered")

            fallback_transcript = self._try_fallbacks(audio, params)

            # Compare primary best vs fallback best
            if fallback_transcript.confidence > best_transcript.confidence:
                logger.info(
                    f"Fallback engine '{fallback_transcript.engine_name}' "
                    f"produced better result "
                    f"({fallback_transcript.confidence:.3f} > "
                    f"{best_transcript.confidence:.3f})"
                )
                best_transcript = fallback_transcript
            else:
                logger.debug("Primary engine still best after fallback attempt")

        # 5. Record metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        metrics.histogram("stt.orchestrator_latency_ms", elapsed_ms)
        metrics.gauge("stt.last_confidence", best_transcript.confidence)

        logger.debug(
            f"Orchestrator result: '{best_transcript.text}' "
            f"(conf={best_transcript.confidence:.3f}, "
            f"engine={best_transcript.engine_name}, "
            f"latency={elapsed_ms:.0f}ms)"
        )

        return best_transcript

    def _try_fallbacks(
        self,
        audio: AudioFrame,
        params: DecodingParams,
    ) -> Transcript:
        """Try fallback engines in priority order and return the best transcript."""
        fallbacks = self._registry.get_fallbacks()

        if not fallbacks:
            logger.warning("No fallback engines available")
            return Transcript()

        best: Optional[Transcript] = None

        for engine in fallbacks:
            health = self._registry.get_health(engine.name)

            try:
                # Lazy-load fallback engine
                if not engine.is_loaded:
                    logger.info(f"Lazy-loading fallback engine '{engine.name}'")
                    engine.load()

                transcript = engine.transcribe(audio, params)

                if health:
                    health.record_success()

                logger.debug(
                    f"Fallback '{engine.name}': '{transcript.text}' "
                    f"(conf={transcript.confidence:.3f})"
                )

                if best is None or transcript.confidence > best.confidence:
                    best = transcript

                # If we get a high-confidence result, no need to try more engines
                if transcript.confidence >= self._high_threshold:
                    break

            except Exception as e:
                logger.warning(f"Fallback engine '{engine.name}' failed: {e}")
                if health:
                    health.record_failure()
                metrics.counter("stt.fallback_engine_error")

        return best or Transcript()
