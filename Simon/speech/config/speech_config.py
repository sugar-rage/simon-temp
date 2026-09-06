"""
Typed configuration dataclasses for the SIMON speech subsystem.

Every configurable parameter is represented as a typed field in a Pydantic-style
dataclass, with defaults, validation, and documentation.  These are populated
from YAML files by ``loader.py``.

Design decision: We use plain ``dataclasses`` with manual validation rather
than requiring Pydantic as a hard dependency.  This keeps the dependency
footprint minimal for edge deployment while still providing type safety
and sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Configuration for the audio I/O layer."""

    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 32
    buffer_size: int = 500
    preferred_device_id: Optional[int] = None
    device_poll_interval_s: float = 3.0

    # DSP pipeline
    enable_noise_suppression: bool = True
    enable_agc: bool = True
    enable_echo_cancellation: bool = False
    noise_suppression_backend: str = "deepfilternet"  # or "spectral"
    agc_target_rms: float = 3000.0
    agc_min_gain: float = 0.5
    agc_max_gain: float = 10.0

    def __post_init__(self):
        assert 8000 <= self.sample_rate <= 48000, f"Sample rate out of range: {self.sample_rate}"
        assert self.channels in (1, 2), f"Unsupported channel count: {self.channels}"
        assert 10 <= self.frame_duration_ms <= 100, f"Frame duration out of range: {self.frame_duration_ms}"


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    backend: str = "silero"  # "silero" or "energy"
    speech_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 700
    max_speech_duration_s: float = 30.0
    padding_duration_ms: int = 300
    window_size_samples: int = 512  # Silero VAD internal window

    def __post_init__(self):
        assert 0.0 < self.speech_threshold < 1.0, f"Speech threshold out of range: {self.speech_threshold}"


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""

    backend: str = "openwakeword"  # "openwakeword" or "text_match"
    wake_word: str = "simon"
    detection_threshold: float = 0.7
    model_path: Optional[str] = None  # None = use default bundled model
    cooldown_s: float = 1.0  # Minimum time between activations
    always_listen: bool = False  # If True, bypass wake word (always active)


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""

    # Engine
    engine: str = "faster-whisper"
    model_size: str = "medium"
    compute_type: str = "float16"  # "float16", "int8", "float32"
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Decoding defaults (overridden by adaptive decoder per scene)
    beam_size: int = 5
    temperature: float = 0.0
    patience: float = 1.0
    best_of: int = 1
    language: str = "en"

    # Confidence thresholds
    default_confidence_threshold: float = 0.60
    safety_confidence_threshold: float = 0.85
    navigation_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.50

    # Whisper-specific
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    word_timestamps: bool = True
    vad_filter: bool = True  # faster-whisper's built-in VAD filter

    # Timeouts
    transcription_timeout_s: float = 15.0

    def __post_init__(self):
        assert self.beam_size >= 1, f"Beam size must be >= 1: {self.beam_size}"
        assert 0.0 <= self.temperature <= 1.0, f"Temperature out of range: {self.temperature}"


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech (used in Phase 2, defined here for config completeness)."""

    engine: str = "pyttsx3"  # "neural", "pyttsx3"
    neural_model: str = "xtts-v2"
    voice_id: Optional[str] = None
    speech_rate: int = 175  # Words per minute for pyttsx3
    volume: float = 0.9
    cache_enabled: bool = True
    cache_max_entries: int = 500
    streaming_enabled: bool = True
    streaming_chunk_size: int = 1  # Sentences per chunk


@dataclass
class SceneConfig:
    """Configuration for acoustic scene classification (Phase 3)."""

    enabled: bool = True
    classification_interval_s: float = 3.0
    ema_alpha: float = 0.3
    history_size: int = 5
    # Multi-engine STT
    enable_multi_engine: bool = False
    enable_intelligent_redecoding: bool = True
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.50
    redecode_temperatures: str = "0.2,0.4"  # Comma-separated


@dataclass
class SpeakerConfig:
    """Configuration for speaker verification (used in Phase 5, defined here for completeness)."""

    enabled: bool = False
    model: str = "ecapa-tdnn"
    verification_threshold: float = 0.65
    enrollment_min_samples: int = 3
    enrollment_max_samples: int = 10
    profiles_dir: str = "data/speaker_profiles"


@dataclass
class MonitoringConfig:
    """Configuration for logging and metrics."""

    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_enabled: bool = True
    dashboard_enabled: bool = False
    dashboard_mode: str = "tui"  # "tui" or "web"
    dashboard_port: int = 8765
    health_check_interval_s: float = 30.0


@dataclass
class SpeechConfig:
    """Top-level configuration for the entire speech subsystem.

    This is the single root config object passed to ``SpeechManager``.
    Sub-configs are accessed as ``config.audio``, ``config.stt``, etc.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Command map — maps normalised voice commands to action names.
    # Loaded from speech.yaml.  This replaces the hardcoded COMMAND_MAP
    # in the legacy speech_input.py.
    command_map: dict = field(default_factory=lambda: {
        "navigate": ["navigate to", "take me to", "go to", "directions to"],
        "read_text": ["read text", "read this", "read sign", "what does it say", "read"],
        "save_face": ["save face", "remember face", "save this face"],
        "identify_face": ["who is this", "identify face", "who is that"],
        "describe": ["describe", "what do you see", "what is around me", "describe surroundings"],
        "status": ["status", "system status", "what is happening"],
        "stop": ["stop", "shut up", "be quiet", "silence"],
        "cancel_nav": ["cancel navigation", "cancel", "stop navigation"],
        "toggle_indoor": ["indoor mode", "outdoor mode", "toggle indoor", "toggle outdoor"],
        "help": ["help", "what can you do", "commands"],
        "repeat": ["repeat", "say again", "what did you say"],
        "battery": ["battery", "battery level", "power"],
        "time": ["what time is it", "time", "current time"],
        "volume_up": ["volume up", "louder", "speak louder"],
        "volume_down": ["volume down", "softer", "speak softer"],
        # Face registration responses
        "yes": ["yes", "yeah", "yep", "sure", "okay", "ok", "save them", "save this person", "save him", "save her"],
        "no": ["no", "nope", "no thanks", "cancel", "don't save", "do not save", "don't save them"],
        "no_name": ["no name", "don't give a name", "save without a name", "without a name"],
    })
