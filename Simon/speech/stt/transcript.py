"""
Transcript data model — rich output from an STT engine.

Replaces the simple string returned by the legacy Whisper wrapper with
a structured object carrying per-word timestamps, confidence scores,
language detection results, and engine provenance metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordInfo:
    """A single recognised word with timing and confidence.

    Attributes:
        word:       The recognised word text.
        start:      Start time in seconds relative to audio start.
        end:        End time in seconds relative to audio start.
        confidence: Per-word confidence score [0.0, 1.0] (if available).
    """

    word: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    """Complete transcription result from an STT engine.

    This is the primary output of :meth:`STTEngine.transcribe` and carries
    all metadata needed for downstream processing (confidence gating,
    hallucination filtering, intent parsing, logging, benchmarking).

    Attributes:
        text:              Full transcribed text (stripped, lowercased).
        raw_text:          Original text as returned by the engine (pre-normalisation).
        confidence:        Overall confidence score [0.0, 1.0].
                           Derived from avg_log_prob for Whisper-based engines.
        language:          Detected or forced language code (e.g. "en").
        language_confidence: Confidence of language detection [0.0, 1.0].
        words:             Per-word information (timing, confidence).
        duration_s:        Duration of the audio that was transcribed.
        latency_s:         Wall-clock time taken for transcription.
        engine_name:       Name of the STT engine that produced this transcript.
        temperature:       The decoding temperature used.
        avg_log_prob:      Average log probability of the decoded tokens.
        compression_ratio: Text compression ratio (indicator of repetition/hallucination).
        no_speech_prob:    Probability that the audio contains no speech.
        is_partial:        True if this is a streaming partial result (not final).
        timestamp:         Monotonic timestamp when this transcript was created.
    """

    text: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    language: str = "en"
    language_confidence: float = 0.0
    words: list[WordInfo] = field(default_factory=list)
    duration_s: float = 0.0
    latency_s: float = 0.0
    engine_name: str = ""
    temperature: float = 0.0
    avg_log_prob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    is_partial: bool = False
    timestamp: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        """True if the transcript contains no meaningful text."""
        return not self.text or not self.text.strip()

    @property
    def word_count(self) -> int:
        """Number of words in the transcript."""
        return len(self.text.split()) if self.text else 0

    @property
    def has_word_timestamps(self) -> bool:
        """True if per-word timing information is available."""
        return len(self.words) > 0

    @property
    def real_time_factor(self) -> float:
        """Ratio of processing time to audio duration.

        RTF < 1.0 means faster-than-realtime.  Lower is better.
        """
        if self.duration_s <= 0:
            return 0.0
        return self.latency_s / self.duration_s

    @property
    def is_likely_hallucination(self) -> bool:
        """Heuristic check for common Whisper hallucination indicators.

        A transcript is flagged as suspicious if:
        - compression_ratio is very high (repetitive text)
        - avg_log_prob is very low (uncertain decoding)
        - no_speech_prob is high but text was produced

        This is a quick pre-filter; the full HallucinationFilter does
        more thorough checking.
        """
        if self.compression_ratio > 2.4 and self.avg_log_prob < -1.0:
            return True
        if self.no_speech_prob > 0.7 and self.text.strip():
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Confidence estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def logprob_to_confidence(avg_log_prob: float) -> float:
        """Convert average log probability to a 0–1 confidence score.

        Uses a sigmoid-like mapping calibrated for Whisper's typical
        output range:
            - avg_log_prob ≈ -0.2  →  confidence ≈ 0.95
            - avg_log_prob ≈ -0.7  →  confidence ≈ 0.70
            - avg_log_prob ≈ -1.5  →  confidence ≈ 0.30

        This is more useful for threshold comparisons than raw log probs.
        """
        import math
        # Shift and scale so that typical Whisper values map to [0, 1]
        return 1.0 / (1.0 + math.exp(-3.0 * (avg_log_prob + 0.5)))

    # ------------------------------------------------------------------ #
    #  Comparison for multi-engine orchestration
    # ------------------------------------------------------------------ #

    def is_better_than(self, other: Transcript) -> bool:
        """Compare two transcripts and return True if *self* is better.

        Used by the engine orchestrator and intelligent redecoder to
        select the best transcript across multiple decoding passes.

        Comparison criteria (in order):
        1. Higher confidence wins.
        2. On tie: lower compression_ratio wins (less repetitive).
        3. On tie: higher avg_log_prob wins.
        """
        if self.confidence != other.confidence:
            return self.confidence > other.confidence
        if self.compression_ratio != other.compression_ratio:
            return self.compression_ratio < other.compression_ratio
        return self.avg_log_prob > other.avg_log_prob

    # ------------------------------------------------------------------ #
    #  Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        parts = [f"text={self.text[:40]!r}"]
        if len(self.text) > 40:
            parts[0] += "..."
        parts.append(f"conf={self.confidence:.2f}")
        parts.append(f"lang={self.language}")
        if self.engine_name:
            parts.append(f"engine={self.engine_name}")
        parts.append(f"rtf={self.real_time_factor:.2f}")
        return f"Transcript({', '.join(parts)})"
