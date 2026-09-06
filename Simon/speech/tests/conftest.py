"""
Shared test fixtures and mocks for the SIMON speech test suite.

Provides reusable fixtures for:
- Synthetic audio frames (silence, sine waves, speech-like signals)
- Mock STT engines, VAD, wake word detectors
- Default test configurations
- Helper functions for creating test data
"""

from __future__ import annotations

import numpy as np
import pytest

from speech.config.speech_config import (
    AudioConfig,
    MonitoringConfig,
    SpeechConfig,
    STTConfig,
    VADConfig,
    WakeWordConfig,
)
from speech.models.audio_frame import AudioFrame
from speech.models.decoding_params import DecodingParams
from speech.models.speech_priority import SpeechPriority
from speech.stt.transcript import Transcript, WordInfo


# ---------------------------------------------------------------------- #
#  Audio frame factories
# ---------------------------------------------------------------------- #

def make_silence_frame(
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    dtype: str = "int16",
) -> AudioFrame:
    """Create a silent audio frame."""
    num_samples = int(sample_rate * duration_s)
    if dtype == "int16":
        data = np.zeros(num_samples, dtype=np.int16)
    else:
        data = np.zeros(num_samples, dtype=np.float32)
    return AudioFrame(data=data, sample_rate=sample_rate, dtype=dtype)


def make_sine_frame(
    frequency: float = 440.0,
    duration_s: float = 0.5,
    amplitude: float = 0.5,
    sample_rate: int = 16000,
    dtype: str = "int16",
) -> AudioFrame:
    """Create an audio frame with a sine wave (simulates speech energy)."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)

    if dtype == "int16":
        data = (signal * 32767).astype(np.int16)
    else:
        data = signal.astype(np.float32)

    return AudioFrame(data=data, sample_rate=sample_rate, dtype=dtype)


def make_noise_frame(
    duration_s: float = 0.5,
    amplitude: float = 0.1,
    sample_rate: int = 16000,
    dtype: str = "int16",
) -> AudioFrame:
    """Create an audio frame with random noise."""
    num_samples = int(sample_rate * duration_s)
    rng = np.random.default_rng(42)
    signal = rng.uniform(-amplitude, amplitude, num_samples)

    if dtype == "int16":
        data = (signal * 32767).astype(np.int16)
    else:
        data = signal.astype(np.float32)

    return AudioFrame(data=data, sample_rate=sample_rate, dtype=dtype)


# ---------------------------------------------------------------------- #
#  Fixtures
# ---------------------------------------------------------------------- #

@pytest.fixture
def silence_frame():
    """A 0.5s silence frame at 16kHz int16."""
    return make_silence_frame()


@pytest.fixture
def speech_frame():
    """A 0.5s sine wave frame simulating speech energy."""
    return make_sine_frame(frequency=300, amplitude=0.6)


@pytest.fixture
def noise_frame():
    """A 0.5s noise frame with low amplitude."""
    return make_noise_frame(amplitude=0.05)


@pytest.fixture
def long_speech_frame():
    """A 3s sine wave simulating a longer speech segment."""
    return make_sine_frame(frequency=300, amplitude=0.6, duration_s=3.0)


@pytest.fixture
def test_config():
    """A test SpeechConfig with fast, lightweight settings."""
    return SpeechConfig(
        audio=AudioConfig(
            sample_rate=16000,
            frame_duration_ms=30,
            buffer_size=100,
            enable_noise_suppression=False,
            enable_agc=False,
        ),
        vad=VADConfig(
            backend="energy",
            speech_threshold=0.3,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        ),
        wakeword=WakeWordConfig(
            backend="text_match",
            wake_word="simon",
            always_listen=False,
        ),
        stt=STTConfig(
            engine="faster-whisper",
            model_size="tiny",
            device="cpu",
            compute_type="float32",
            default_confidence_threshold=0.50,
        ),
        monitoring=MonitoringConfig(
            log_level="DEBUG",
            metrics_enabled=True,
        ),
    )


@pytest.fixture
def sample_transcript():
    """A sample Transcript for testing postprocessing."""
    return Transcript(
        text="navigate to the library",
        raw_text="Navigate to the library",
        confidence=0.92,
        language="en",
        words=[
            WordInfo(word="navigate", start=0.0, end=0.4, confidence=0.95),
            WordInfo(word="to", start=0.4, end=0.5, confidence=0.98),
            WordInfo(word="the", start=0.5, end=0.6, confidence=0.97),
            WordInfo(word="library", start=0.6, end=1.1, confidence=0.88),
        ],
        duration_s=1.1,
        latency_s=0.15,
        engine_name="faster-whisper",
        avg_log_prob=-0.25,
        compression_ratio=1.3,
        no_speech_prob=0.02,
    )


@pytest.fixture
def low_confidence_transcript():
    """A transcript with low confidence for gate testing."""
    return Transcript(
        text="navigate to the highway",
        raw_text="Navigate to the highway",
        confidence=0.45,
        language="en",
        duration_s=1.0,
        latency_s=0.2,
        engine_name="faster-whisper",
        avg_log_prob=-1.5,
        compression_ratio=1.8,
        no_speech_prob=0.1,
    )


@pytest.fixture
def hallucination_transcript():
    """A transcript exhibiting hallucination patterns."""
    return Transcript(
        text="thank you for watching subscribe to my channel",
        raw_text="Thank you for watching subscribe to my channel",
        confidence=0.30,
        language="en",
        duration_s=0.5,
        latency_s=0.1,
        engine_name="faster-whisper",
        avg_log_prob=-2.0,
        compression_ratio=3.5,
        no_speech_prob=0.8,
    )
