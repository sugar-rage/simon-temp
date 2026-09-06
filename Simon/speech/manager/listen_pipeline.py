"""
Listen pipeline — orchestrates the full STT pipeline from audio to command.

Pipeline stages:
    AudioStream → DSP → VAD → [WakeWord] → STT → HallucinationFilter →
    ConfidenceGate → IntentParser → SpeechCommand

This module owns the listen loop: it reads frames from the audio stream,
processes them through each stage, and emits SpeechCommand objects.

Design decision: The pipeline runs in a dedicated background thread.
Commands are placed into a thread-safe queue for the SpeechManager to
consume.  This decouples audio processing from command handling.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from speech.audio.audio_stream import AudioStream
from speech.audio.dsp_pipeline import DSPPipeline
from speech.config.speech_config import SpeechConfig
from speech.errors.exceptions import SpeechError
from speech.models.audio_frame import AudioFrame
from speech.models.speech_command import CommandSource, CommandStatus, SpeechCommand
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector
from speech.postprocessing.hallucination_filter import HallucinationFilter
from speech.stt.base import BaseSTTEngine
from speech.stt.confidence_gate import ConfidenceGate
from speech.stt.transcript import Transcript
from speech.vad.base import BaseVAD
from speech.wakeword.base import BaseWakeWordDetector
from speech.wakeword.text_match_detector import TextMatchDetector

logger = get_logger("manager.listen_pipeline")
metrics = get_collector()


class ListenPipeline:
    """Orchestrates the full audio → command pipeline.

    Args:
        audio_stream:      Audio input stream.
        dsp_pipeline:      DSP processing chain.
        vad:               Voice Activity Detector.
        wake_word_detector: Wake word detector.
        stt_engine:        Primary STT engine.
        confidence_gate:   Confidence threshold enforcer.
        hallucination_filter: Hallucination pattern detector.
        config:            Speech configuration.
        command_map:       Action name → trigger phrases mapping.
    """

    def __init__(
        self,
        audio_stream: AudioStream,
        dsp_pipeline: DSPPipeline,
        vad: BaseVAD,
        wake_word_detector: BaseWakeWordDetector,
        stt_engine: BaseSTTEngine,
        confidence_gate: ConfidenceGate,
        hallucination_filter: HallucinationFilter,
        config: SpeechConfig,
        command_map: Optional[dict[str, list[str]]] = None,
    ):
        self._stream = audio_stream
        self._dsp = dsp_pipeline
        self._vad = vad
        self._wakeword = wake_word_detector
        self._stt = stt_engine
        self._confidence_gate = confidence_gate
        self._hallucination_filter = hallucination_filter
        self._config = config
        self._command_map = command_map or config.command_map

        # Output queue
        self._command_queue: queue.Queue[SpeechCommand] = queue.Queue(maxsize=50)

        # State
        self._running = False
        self._paused = False
        self._listen_thread: Optional[threading.Thread] = None
        self._wake_word_active = not config.wakeword.always_listen

        # Text-match fallback reference (for wake-word stripping)
        self._text_matcher = (
            wake_word_detector
            if isinstance(wake_word_detector, TextMatchDetector)
            else TextMatchDetector(config.wakeword.wake_word)
        )

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the listen pipeline in a background thread."""
        if self._running:
            return

        # Ensure STT model is loaded
        if not self._stt.is_loaded:
            self._stt.load()

        self._running = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="listen-pipeline"
        )
        self._listen_thread.start()
        logger.info("Listen pipeline started")

    def stop(self) -> None:
        """Stop the listen pipeline."""
        self._running = False
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=5.0)
        logger.info("Listen pipeline stopped")

    def pause(self) -> None:
        """Pause listening (ignore incoming audio)."""
        self._paused = True
        if self._stream:
            self._stream.pause()
        if self._vad:
            self._vad.reset()

    def resume(self) -> None:
        """Resume listening."""
        self._paused = False
        self._just_resumed = True
        if self._stream:
            self._stream.resume()
        if self._vad:
            self._vad.reset()

    def set_expecting_name(self, expecting: bool) -> None:
        """Set whether the pipeline is expecting a spoken name."""
        logger.info(
            "[NAME_TRACE] ListenPipeline set_expecting_name(%s) called (has_gate=%s)",
            expecting, (self._confidence_gate is not None),
        )
        if self._confidence_gate and hasattr(self._confidence_gate, "set_expecting_name"):
            self._confidence_gate.set_expecting_name(expecting)
        logger.info("[NAME_TRACE] ListenPipeline expecting_name=%s", expecting)

    def set_expecting_confirmation(self, expecting: bool) -> None:
        """Set whether the pipeline is expecting a confirmation (yes/no) response."""
        logger.info(
            "[CONFIRM_TRACE] ListenPipeline set_expecting_confirmation(%s) called (has_gate=%s)",
            expecting, (self._confidence_gate is not None),
        )
        if self._confidence_gate and hasattr(self._confidence_gate, "set_expecting_confirmation"):
            self._confidence_gate.set_expecting_confirmation(expecting)
        logger.info("[CONFIRM_TRACE] ListenPipeline expecting_confirmation=%s", expecting)

    # ------------------------------------------------------------------ #
    #  Command retrieval
    # ------------------------------------------------------------------ #

    def get_command(self, timeout: float = 0.5) -> Optional[SpeechCommand]:
        """Retrieve the next parsed command from the pipeline.

        Args:
            timeout: Max seconds to wait for a command.

        Returns:
            A ``SpeechCommand``, or ``None`` if timeout expired.
        """
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------ #
    #  Main listen loop
    # ------------------------------------------------------------------ #

    def _listen_loop(self) -> None:
        """Background thread: read audio → DSP → VAD → STT → parse."""
        logger.info("Listen loop running")

        while self._running:
            try:
                # Read a frame from the audio stream
                frame = self._stream.read_frame(timeout=0.1)
                if self._paused:
                    continue
                if frame is None:
                    continue

                if getattr(self, '_just_resumed', False):
                    logger.info("[STT] listening started")
                    self._just_resumed = False

                # DSP processing
                processed = self._dsp.process(frame)

                # VAD
                vad_result = self._vad.process_frame(processed)

                # Check for completed speech segment
                speech_segment = self._vad.get_speech_segment()
                if speech_segment is None:
                    continue

                # We have a complete speech segment — process it
                self._process_speech_segment(speech_segment)

            except SpeechError as e:
                logger.error(f"Listen pipeline error: {e}")
                metrics.counter("pipeline.listen_errors")
                if not e.recoverable:
                    logger.error("Non-recoverable error in listen pipeline")
                    break
            except Exception as e:
                logger.error(f"Unexpected listen pipeline error: {e}")
                metrics.counter("pipeline.listen_errors")
                time.sleep(0.1)  # Brief pause to prevent tight error loop

    def _process_speech_segment(self, segment: AudioFrame) -> None:
        """Process a complete speech segment through STT and parsing.

        Args:
            segment: The accumulated speech audio from VAD.
        """
        t0 = time.monotonic()

        # Transcribe
        try:
            transcript = self._stt.transcribe(segment)
        except SpeechError as e:
            logger.error(f"STT failed: {e}")
            metrics.counter("stt.errors")
            return

        # Hallucination filter
        transcript = self._hallucination_filter.filter(transcript)
        if transcript is None:
            return

        # Empty text check
        if transcript.is_empty:
            return

        text = transcript.text.strip()

        # Wake word check / stripping
        if self._wake_word_active:
            ww_result = self._text_matcher.check_transcript(text)
            if not ww_result.detected:
                logger.debug(f"No wake word in: '{text[:30]}'")
                return
            # Strip wake word from text
            text = self._text_matcher.strip_wake_word(text)
            if not text:
                # Just the wake word with nothing after it
                metrics.counter("wakeword.solo_activations")
                return
            metrics.counter("wakeword.detections")
        else:
            # When always_listen is true, if wake word was spoken, strip it cleanly
            text = self._text_matcher.strip_wake_word(text)
            if not text:
                return

        logger.info("[SPEECH] STT transcription: %r (confidence=%.2f)", text, transcript.confidence)
        logger.info("[NAME_TRACE] STT transcript: %r (raw=%r, confidence=%.2f)", text, transcript.raw_text, transcript.confidence)
        logger.info("[NAME_TRACE] confidence: %.2f", transcript.confidence)
        logger.info("[NAME_TRACE] expecting_name: %s", getattr(self._confidence_gate, "expecting_name", False))
        logger.info("[CONFIRM_TRACE] STT transcript: %r (raw=%r, confidence=%.2f)", text, transcript.raw_text, transcript.confidence)
        logger.info("[CONFIRM_TRACE] confidence: %.2f", transcript.confidence)
        logger.info("[CONFIRM_TRACE] expecting_confirmation: %s", getattr(self._confidence_gate, "expecting_confirmation", False))

        # Intent parsing (with punctuation normalization)
        action, args = self._parse_intent(text)
        if not action:
            logger.debug(f"No matching command for: '{text[:40]}'")
            metrics.counter("pipeline.unmatched_commands")
            return

        logger.info("[NAME_TRACE] parsed action: %r, args: %r", action, args)
        logger.info("[CONFIRM_TRACE] parsed action: %r, args: %r", action, args)

        # Per-action confidence check (evaluated using the specific action's threshold)
        action_gate = self._confidence_gate.check(transcript, action=action)
        logger.info("[NAME_TRACE] confidence gate: passed=%s, action=%r, threshold=%.2f, conf=%.2f, reason=%r",
                    action_gate.passed, action_gate.action, action_gate.threshold, action_gate.confidence, action_gate.reason)
        logger.info("[CONFIRM_TRACE] confidence gate: passed=%s, action=%r, threshold=%.2f, conf=%.2f, reason=%r",
                    action_gate.passed, action_gate.action, action_gate.threshold, action_gate.confidence, action_gate.reason)
        if not action_gate.passed:
            logger.info(f"[SPEECH] Confidence gate blocked: {action_gate.reason}")
            return

        # Build command
        status = CommandStatus.APPROVED

        command = SpeechCommand(
            action=action,
            args=args,
            confidence=transcript.confidence,
            raw_transcript=transcript.raw_text,
            source=CommandSource.VOICE,
            status=status,
            engine_name=transcript.engine_name,
            decoding_passes=1,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        metrics.histogram("pipeline.listen_latency_ms", elapsed_ms)

        logger.info("[SPEECH] SpeechCommand generated: action=%r, args=%r, conf=%.2f", action, args, transcript.confidence)
        logger.info("[NAME_TRACE] SpeechCommand: action=%r, args=%r, conf=%.2f, raw=%r", action, args, transcript.confidence, transcript.raw_text)

        # Queue the command
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            logger.warning("Command queue full, dropping oldest")
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            self._command_queue.put_nowait(command)

    # ------------------------------------------------------------------ #
    #  Intent parsing
    # ------------------------------------------------------------------ #

    def _parse_intent(self, text: str) -> tuple[str, str]:
        """Parse normalised text into (action, args).

        Matches the text against the command map using longest-match-first
        strategy with word-boundary and punctuation checks.

        Args:
            text: Normalised, wake-word-stripped text.

        Returns:
            Tuple of (action_name, extracted_args).  Returns ("", "")
            if no match is found.
        """
        import re
        import string

        # Normalize text and strip punctuation
        text_clean = text.lower().strip().strip(string.punctuation).strip()
        text_norm = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        text_norm = re.sub(r'\s+', ' ', text_norm)

        # Sort phrases by length (longest first) to match most specific
        matches: list[tuple[str, str, int]] = []  # (action, args, phrase_len)

        for action, phrases in self._command_map.items():
            action_str = "yes" if action is True else ("no" if action is False else str(action))
            if isinstance(phrases, str):
                phrases = [phrases]
            for phrase in phrases:
                phrase_str = "yes" if phrase is True else ("no" if phrase is False else str(phrase))
                phrase_lower = phrase_str.lower().strip()
                if text_clean == phrase_lower or text_norm == phrase_lower:
                    matches.append((action_str, "", len(phrase_lower)))
                elif text_clean.startswith(phrase_lower + " ") or text_norm.startswith(phrase_lower + " "):
                    arg_src = text_clean if text_clean.startswith(phrase_lower + " ") else text_norm
                    args = arg_src[len(phrase_lower) + 1:].strip()
                    matches.append((action_str, args, len(phrase_lower)))

        if not matches:
            cleaned_text = text.strip().strip(string.punctuation).strip()
            return ("unmatched_text", cleaned_text)

        # Select the longest matching phrase (most specific)
        matches.sort(key=lambda m: m[2], reverse=True)
        best_action, best_args, _ = matches[0]
        return (best_action, best_args)
