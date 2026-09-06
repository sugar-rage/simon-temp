"""
SIMON Speech Subsystem — Production-grade speech I/O for assistive navigation.

This package provides:
    - Audio capture with DSP (noise suppression, AGC, echo cancellation)
    - Voice Activity Detection (Silero VAD)
    - Wake word detection (OpenWakeWord)
    - Speech-to-Text (faster-whisper with multi-engine orchestration)
    - Text-to-Speech (neural TTS with emotional profiles)
    - Priority speech queue with preemption
    - Safety validation layer
    - Conversational context memory
    - Personal adaptation

Entry point:
    from speech.manager.speech_manager import SpeechManager
"""

__version__ = "2.0.0"
