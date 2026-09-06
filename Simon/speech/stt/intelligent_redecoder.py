"""
Intelligent Redecoder — multi-temperature retry strategy for STT.

When the primary STT decode at temperature=0.0 returns low confidence,
the redecoder runs additional passes at higher temperatures to explore
alternative decodings. It selects the best transcript by comparing
avg_log_prob, compression_ratio, and consensus across passes.

This only triggers on low-confidence utterances, so the amortized
overhead is <10% (most speech is clear).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.monitoring.metrics import get_collector
from speech.stt.base import BaseSTTEngine
from speech.stt.transcript import Transcript

logger = logging.getLogger(__name__)
metrics = get_collector()


class IntelligentRedecoder:
    """Multi-temperature redecoding strategy.

    Args:
        high_confidence_threshold: If primary confidence >= this, skip redecoding.
        low_confidence_threshold: If best confidence after redecoding < this,
                                   signal the orchestrator to try a fallback engine.
        retry_temperatures: Temperatures to try for redecoding (in order).
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.50,
        retry_temperatures: Optional[List[float]] = None,
    ):
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold
        self._retry_temps = retry_temperatures or [0.2, 0.4]

    def decode_with_retry(
        self,
        engine: BaseSTTEngine,
        audio: AudioFrame,
        params: DecodingParams,
    ) -> RedecoderResult:
        """Run the primary decode and optionally retry at higher temperatures.

        Args:
            engine: The STT engine to use for decoding.
            audio: Audio to transcribe.
            params: Base decoding parameters (temperature will be overridden).

        Returns:
            A RedecoderResult with the best transcript and metadata.
        """
        # 1. Primary decode at temperature=0.0
        primary_params = params.with_temperature(0.0)
        primary_transcript = engine.transcribe(audio, primary_params)
        all_transcripts = [primary_transcript]

        logger.debug(
            f"Primary decode: '{primary_transcript.text}' "
            f"(conf={primary_transcript.confidence:.3f})"
        )

        # 2. If high confidence, return immediately
        if primary_transcript.confidence >= self._high_threshold:
            return RedecoderResult(
                best=primary_transcript,
                all_transcripts=all_transcripts,
                retries_used=0,
                needs_fallback=False,
            )

        # 3. Low/medium confidence: retry at higher temperatures
        metrics.counter("stt.redecode_triggered")

        for temp in self._retry_temps:
            retry_params = params.with_temperature(temp)
            retry_transcript = engine.transcribe(audio, retry_params)
            all_transcripts.append(retry_transcript)

            logger.debug(
                f"Retry at temp={temp}: '{retry_transcript.text}' "
                f"(conf={retry_transcript.confidence:.3f})"
            )

        # 4. Select the best transcript
        best = self._select_best(all_transcripts)

        # 5. Check if we still need fallback
        needs_fallback = best.confidence < self._low_threshold

        if needs_fallback:
            logger.info(
                f"Redecoding could not reach acceptable confidence "
                f"({best.confidence:.3f} < {self._low_threshold}), "
                f"recommending fallback engine"
            )

        return RedecoderResult(
            best=best,
            all_transcripts=all_transcripts,
            retries_used=len(self._retry_temps),
            needs_fallback=needs_fallback,
        )

    def _select_best(self, transcripts: List[Transcript]) -> Transcript:
        """Select the best transcript from multiple decoding passes.

        Strategy:
        1. If 2+ transcripts agree on the text, use the consensus (highest confidence).
        2. Otherwise, select by highest avg_log_prob.
        3. Break ties with lowest compression_ratio.
        """
        if not transcripts:
            return Transcript()

        if len(transcripts) == 1:
            return transcripts[0]

        # Check for consensus (2+ agreeing texts)
        text_groups: dict[str, List[Transcript]] = {}
        for t in transcripts:
            normalized = t.text.strip().lower()
            if normalized not in text_groups:
                text_groups[normalized] = []
            text_groups[normalized].append(t)

        # Find largest consensus group
        largest_group = max(text_groups.values(), key=len)
        if len(largest_group) >= 2:
            # Consensus found: return the one with highest confidence
            return max(largest_group, key=lambda t: t.confidence)

        # No consensus: select by confidence (derived from avg_log_prob)
        # Use avg_log_prob as primary, compression_ratio as tiebreaker
        def score(t: Transcript) -> tuple:
            return (t.confidence, -t.compression_ratio)

        return max(transcripts, key=score)


class RedecoderResult:
    """Result of the intelligent redecoding process.

    Attributes:
        best: The best transcript selected from all passes.
        all_transcripts: All transcripts generated during redecoding.
        retries_used: Number of retry passes performed.
        needs_fallback: True if confidence is still below the low threshold,
                         indicating the orchestrator should try a fallback engine.
    """

    __slots__ = ("best", "all_transcripts", "retries_used", "needs_fallback")

    def __init__(
        self,
        best: Transcript,
        all_transcripts: List[Transcript],
        retries_used: int,
        needs_fallback: bool,
    ):
        self.best = best
        self.all_transcripts = all_transcripts
        self.retries_used = retries_used
        self.needs_fallback = needs_fallback
