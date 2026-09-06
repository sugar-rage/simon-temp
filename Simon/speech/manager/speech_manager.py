"""
SpeechManager — the unified facade for the SIMON speech subsystem.

Replaces the legacy ``SpeechInput`` + ``VoiceEngine`` pair with a single
entry point that wires up all components, manages their lifecycle,
and provides the public API consumed by ``main.py``.

Public API:
    manager = SpeechManager(config)
    manager.start()
    command = manager.get_command()       # returns SpeechCommand or None
    manager.speak("Turn left in 50 meters", priority=SpeechPriority.NAVIGATION)
    manager.stop()

Backward compatibility:
    command["action"]  →  works (SpeechCommand supports dict-style access)
    command["args"]    →  works
    command.get("action", "")  →  works
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from speech.audio.audio_stream import AudioStream
from speech.audio.device_manager import DeviceManager
from speech.audio.dsp_pipeline import DSPPipeline
from speech.audio.gain_controller import GainController
from speech.audio.noise_suppressor import NoiseSuppressor
from speech.config.loader import load_config, load_default_config
from speech.config.speech_config import SpeechConfig
from speech.errors.exceptions import DeviceNotFoundError, SpeechError
from speech.errors.recovery import ErrorRecoveryManager, RecoveryAction
from speech.manager.listen_pipeline import ListenPipeline
from speech.models.speech_command import SpeechCommand
from speech.models.speech_priority import SpeechPriority
from speech.monitoring.logger import configure_logging, get_logger
from speech.monitoring.metrics import get_collector
from speech.postprocessing.hallucination_filter import HallucinationFilter
from speech.stt.confidence_gate import ConfidenceGate
from speech.stt.faster_whisper_engine import FasterWhisperEngine
from speech.vad.base import BaseVAD
from speech.vad.energy_vad import EnergyVAD
from speech.vad.silero_vad import SileroVAD
from speech.wakeword.base import BaseWakeWordDetector
from speech.wakeword.openwakeword_detector import OpenWakeWordDetector
from speech.wakeword.text_match_detector import TextMatchDetector

# TTS components
from speech.manager.speak_pipeline import SpeakPipeline
from speech.queue.priority_queue import SpeechPriorityQueue
from speech.tts.speech_cache import SpeechCache
from speech.tts.text_normalizer import TextNormalizer
from speech.tts.pronunciation_dict import PronunciationDictionary
from speech.tts.streaming_synthesizer import StreamingSynthesizer
from speech.tts.neural_tts_engine import NeuralTTSEngine
from speech.tts.pyttsx3_engine import Pyttsx3Engine

logger = get_logger("manager")
metrics = get_collector()


class SpeechManager:
    """Unified facade for the SIMON speech subsystem.

    Wires up all speech components and manages their lifecycle.
    This is the single entry point that ``main.py`` interacts with.

    Args:
        config: SpeechConfig instance.  If None, loads defaults.
        user_config_path: Optional path to a user YAML override file.
    """

    def __init__(
        self,
        config: Optional[SpeechConfig] = None,
        user_config_path: Optional[str] = None,
    ):
        self._config = config or load_config(user_config_path)
        self._started = False
        self._lock = threading.Lock()

        # Components (initialised in start())
        self._device_manager: Optional[DeviceManager] = None
        self._audio_stream: Optional[AudioStream] = None
        self._dsp_pipeline: Optional[DSPPipeline] = None
        self._vad: Optional[BaseVAD] = None
        self._wakeword: Optional[BaseWakeWordDetector] = None
        self._stt_engine: Optional[FasterWhisperEngine] = None
        self._confidence_gate: Optional[ConfidenceGate] = None
        self._hallucination_filter: Optional[HallucinationFilter] = None
        self._listen_pipeline: Optional[ListenPipeline] = None
        self._recovery: Optional[ErrorRecoveryManager] = None

        # TTS Pipeline (Phase 2)
        self._priority_queue: Optional[SpeechPriorityQueue] = None
        self._speak_pipeline: Optional[SpeakPipeline] = None
        self._speak_lock = threading.Lock()

    @property
    def config(self) -> SpeechConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._started

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Initialise all components and start the listen pipeline.

        This is the main entry point.  Calling ``start()`` will:
        1. Configure logging
        2. Discover and connect to the best microphone
        3. Build the DSP pipeline
        4. Initialise VAD, wake word, STT
        5. Start the listen pipeline background thread
        6. Register error recovery strategies
        """
        if self._started:
            logger.warning("SpeechManager already started")
            return

        with self._lock:
            cfg = self._config

            # 1. Logging
            configure_logging(
                level=cfg.monitoring.log_level,
                log_file=cfg.monitoring.log_file,
            )
            logger.info("=" * 60)
            logger.info("SIMON Speech Subsystem v2.0 starting")
            logger.info("=" * 60)

            # 2. Audio device
            self._device_manager = DeviceManager(
                preferred_device_id=cfg.audio.preferred_device_id,
                target_sample_rate=cfg.audio.sample_rate,
                poll_interval_s=cfg.audio.device_poll_interval_s,
            )
            try:
                device = self._device_manager.get_best_device()
            except DeviceNotFoundError:
                logger.error("No microphone found — listen pipeline disabled")
                device = None

            # 3. Audio stream
            if device:
                self._audio_stream = AudioStream(
                    device=device,
                    sample_rate=cfg.audio.sample_rate,
                    channels=cfg.audio.channels,
                    frame_duration_ms=cfg.audio.frame_duration_ms,
                    buffer_size=cfg.audio.buffer_size,
                )

            # 4. DSP pipeline
            noise_suppressor = NoiseSuppressor(
                enable_deep_filter=cfg.audio.enable_noise_suppression
                and cfg.audio.noise_suppression_backend == "deepfilternet",
            )
            gain_controller = GainController(
                target_rms=cfg.audio.agc_target_rms,
                min_gain=cfg.audio.agc_min_gain,
                max_gain=cfg.audio.agc_max_gain,
            )
            self._dsp_pipeline = DSPPipeline.create_default(
                noise_suppressor=noise_suppressor,
                gain_controller=gain_controller,
                enable_noise_suppression=cfg.audio.enable_noise_suppression,
                enable_agc=cfg.audio.enable_agc,
                enable_echo_cancellation=cfg.audio.enable_echo_cancellation,
            )

            # 5. VAD
            self._vad = self._create_vad(cfg)

            # 6. Wake word
            self._wakeword = self._create_wakeword(cfg)

            # 7. STT engine
            self._stt_engine = FasterWhisperEngine(
                model_size=cfg.stt.model_size,
                compute_type=cfg.stt.compute_type,
                device=cfg.stt.device,
                language=cfg.stt.language,
            )

            # 8. Confidence gate
            self._confidence_gate = ConfidenceGate(
                default_threshold=cfg.stt.default_confidence_threshold,
                safety_threshold=cfg.stt.safety_confidence_threshold,
            )

            # 9. Hallucination filter
            self._hallucination_filter = HallucinationFilter(
                max_compression_ratio=cfg.stt.compression_ratio_threshold,
                min_log_prob=cfg.stt.log_prob_threshold,
                max_no_speech_prob=cfg.stt.no_speech_threshold,
            )

            # 10. Recovery manager
            self._recovery = ErrorRecoveryManager()
            self._register_recovery_strategies()

            # 11. Listen pipeline
            if self._audio_stream:
                self._listen_pipeline = ListenPipeline(
                    audio_stream=self._audio_stream,
                    dsp_pipeline=self._dsp_pipeline,
                    vad=self._vad,
                    wake_word_detector=self._wakeword,
                    stt_engine=self._stt_engine,
                    confidence_gate=self._confidence_gate,
                    hallucination_filter=self._hallucination_filter,
                    config=cfg,
                )

                # Start audio stream and listen pipeline
                self._audio_stream.start()
                self._listen_pipeline.start()

                # Start device monitoring
                self._device_manager.start_monitoring()

            # 12. Speak pipeline
            self._priority_queue = SpeechPriorityQueue()
            cache = SpeechCache()
            normalizer = TextNormalizer()
            pronunciation_dict = PronunciationDictionary()
            
            # Select TTS engine based on config
            if cfg.tts.engine == "neural" or cfg.tts.engine == "xtts_v2":
                tts_engine = NeuralTTSEngine(
                    model_name=cfg.tts.neural_model,
                    language=cfg.stt.language,
                )
            else:
                tts_engine = Pyttsx3Engine(
                    rate=cfg.tts.speech_rate,
                    volume=cfg.tts.volume,
                )
                
            synthesizer = StreamingSynthesizer(
                engine=tts_engine,
                cache=cache,
                normalizer=normalizer,
                pronunciation_dict=pronunciation_dict,
            )
            
            self._speak_pipeline = SpeakPipeline(
                manager=self,
                synthesizer=synthesizer,
                priority_queue=self._priority_queue,
            )
            self._speak_pipeline.start()

            self._started = True
            logger.info("SpeechManager started successfully")

    def stop(self) -> None:
        """Stop all components and release resources."""
        with self._lock:
            logger.info("SpeechManager stopping...")

            if self._listen_pipeline:
                self._listen_pipeline.stop()

            if self._speak_pipeline:
                self._speak_pipeline.stop()

            if self._audio_stream:
                self._audio_stream.stop()

            if self._device_manager:
                self._device_manager.stop_monitoring()

            if self._stt_engine and self._stt_engine.is_loaded:
                self._stt_engine.unload()

            self._started = False
            logger.info("SpeechManager stopped")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_command(self, timeout: float = 0.5) -> Optional[SpeechCommand]:
        """Get the next voice command from the listen pipeline.

        This is the primary API for ``main.py``.

        Args:
            timeout: Max seconds to wait.

        Returns:
            A ``SpeechCommand`` with ``.action`` and ``.args``,
            or ``None`` if timeout expired.

        The returned object supports backward-compatible dict access::

            cmd = manager.get_command()
            if cmd:
                action = cmd["action"]  # works
                args = cmd.get("args", "")  # works
        """
        if not self._listen_pipeline:
            return None
        return self._listen_pipeline.get_command(timeout=timeout)

    def speak(
        self,
        text: str,
        priority: SpeechPriority | int = SpeechPriority.NOTIFICATION,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Speak text aloud with the given priority.

        Args:
            text:        The text to speak.
            priority:    Speech priority level.
            on_complete: Optional callback invoked with bool success when playback completes.
        """
        if isinstance(priority, int):
            priority = SpeechPriority.from_legacy(priority)

        logger.info("[SPEECH] TTS queued: %r (priority=%s)", text, priority.name if hasattr(priority, 'name') else priority)

        with self._speak_lock:
            if self._priority_queue:
                self._priority_queue.put(text, priority, on_complete=on_complete)
                metrics.counter("tts.utterances")
            else:
                logger.warning("SpeakPipeline not initialized. Dropping message: " + text)
                if on_complete:
                    try:
                        on_complete(False)
                    except Exception:
                        pass

    def pause_listening(self) -> None:
        """Pause audio capture (e.g. during TTS playback)."""
        if self._audio_stream:
            self._audio_stream.pause()

    def resume_listening(self) -> None:
        """Resume audio capture after pause."""
        if self._audio_stream:
            self._audio_stream.resume()

    def set_expecting_name(self, expecting: bool) -> None:
        """Set whether the system is expecting a spoken name during face registration."""
        logger.info(
            "[NAME_TRACE] SpeechManager set_expecting_name(%s) called (has_gate=%s, has_pipe=%s)",
            expecting, (self._confidence_gate is not None), (self._listen_pipeline is not None),
        )
        if self._confidence_gate and hasattr(self._confidence_gate, "set_expecting_name"):
            self._confidence_gate.set_expecting_name(expecting)
        if self._listen_pipeline and hasattr(self._listen_pipeline, "set_expecting_name"):
            self._listen_pipeline.set_expecting_name(expecting)
        logger.info("[SPEECH] Expecting name context: %s", expecting)
        logger.info("[NAME_TRACE] SpeechManager expecting_name=%s", expecting)

    def set_expecting_confirmation(self, expecting: bool) -> None:
        """Set whether the system is expecting a confirmation (yes/no) response."""
        logger.info(
            "[CONFIRM_TRACE] SpeechManager set_expecting_confirmation(%s) called (has_gate=%s, has_pipe=%s)",
            expecting, (self._confidence_gate is not None), (self._listen_pipeline is not None),
        )
        if self._confidence_gate and hasattr(self._confidence_gate, "set_expecting_confirmation"):
            self._confidence_gate.set_expecting_confirmation(expecting)
        if self._listen_pipeline and hasattr(self._listen_pipeline, "set_expecting_confirmation"):
            self._listen_pipeline.set_expecting_confirmation(expecting)
        logger.info("[SPEECH] Expecting confirmation context: %s", expecting)
        logger.info("[CONFIRM_TRACE] SpeechManager expecting_confirmation=%s", expecting)

    # ------------------------------------------------------------------ #
    #  Component factory methods
    # ------------------------------------------------------------------ #

    def _create_vad(self, cfg: SpeechConfig) -> BaseVAD:
        """Create the appropriate VAD implementation."""
        if cfg.vad.backend == "silero":
            try:
                vad = SileroVAD(
                    speech_threshold=cfg.vad.speech_threshold,
                    min_speech_duration_ms=cfg.vad.min_speech_duration_ms,
                    min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
                    max_speech_duration_s=cfg.vad.max_speech_duration_s,
                    padding_duration_ms=cfg.vad.padding_duration_ms,
                )
                logger.info("Using Silero VAD")
                return vad
            except Exception as e:
                logger.warning(f"Silero VAD failed, falling back to energy: {e}")

        # Fallback
        logger.info("Using energy VAD (fallback)")
        return EnergyVAD(
            min_speech_duration_ms=cfg.vad.min_speech_duration_ms,
            min_silence_duration_ms=cfg.vad.min_silence_duration_ms,
            max_speech_duration_s=cfg.vad.max_speech_duration_s,
        )

    def _create_wakeword(self, cfg: SpeechConfig) -> BaseWakeWordDetector:
        """Create the appropriate wake word detector."""
        if cfg.wakeword.backend == "openwakeword":
            try:
                detector = OpenWakeWordDetector(
                    wake_word=cfg.wakeword.wake_word,
                    threshold=cfg.wakeword.detection_threshold,
                    model_path=cfg.wakeword.model_path,
                    cooldown_s=cfg.wakeword.cooldown_s,
                )
                if detector._model_loaded:
                    logger.info("Using OpenWakeWord detector")
                    return detector
                logger.warning("OpenWakeWord model not loaded, falling back to text match")
            except Exception as e:
                logger.warning(f"OpenWakeWord failed: {e}")

        # Fallback
        logger.info("Using text-match wake word detector (fallback)")
        return TextMatchDetector(wake_word=cfg.wakeword.wake_word)

    # ------------------------------------------------------------------ #
    #  Recovery strategies
    # ------------------------------------------------------------------ #

    def _register_recovery_strategies(self) -> None:
        """Register error recovery strategies for all components."""
        if not self._recovery:
            return

        # Device reconnection
        def reconnect_device() -> bool:
            try:
                if self._device_manager:
                    new_device = self._device_manager.get_best_device()
                    if self._audio_stream:
                        self._audio_stream.stop()
                        self._audio_stream = AudioStream(
                            device=new_device,
                            sample_rate=self._config.audio.sample_rate,
                        )
                        self._audio_stream.start()
                    return True
            except Exception:
                return False

        self._recovery.register(
            DeviceNotFoundError,
            RecoveryAction("reconnect_audio", reconnect_device, max_retries=5),
        )

        # STT model reload
        def reload_stt() -> bool:
            try:
                if self._stt_engine:
                    self._stt_engine.unload()
                    self._stt_engine.load()
                return True
            except Exception:
                return False

        from speech.errors.exceptions import STTTranscriptionError
        self._recovery.register(
            STTTranscriptionError,
            RecoveryAction("reload_stt", reload_stt, max_retries=3),
        )
