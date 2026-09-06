"""
Custom exception hierarchy for the SIMON speech subsystem.

All speech-subsystem exceptions inherit from :class:`SpeechError` so that
callers can catch the entire family with a single ``except SpeechError``.
Sub-hierarchies are organised by component (audio, STT, TTS, etc.) to
enable targeted recovery strategies in ``recovery.py``.

Design decision: Each exception carries a ``recoverable`` flag indicating
whether the error recovery module should attempt automatic recovery.
Non-recoverable errors (e.g. missing model file) propagate to the caller.
"""

from __future__ import annotations


# ====================================================================== #
#  Base
# ====================================================================== #

class SpeechError(Exception):
    """Base exception for all speech subsystem errors."""

    def __init__(self, message: str = "", *, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


# ====================================================================== #
#  Audio / Device errors
# ====================================================================== #

class AudioError(SpeechError):
    """Base for audio I/O and device errors."""
    pass


class DeviceNotFoundError(AudioError):
    """No suitable audio input device was found."""

    def __init__(self, message: str = "No working microphone found"):
        super().__init__(message, recoverable=True)


class DeviceDisconnectedError(AudioError):
    """The active audio device was disconnected during operation."""

    def __init__(self, device_id: int | None = None):
        msg = f"Audio device {device_id} disconnected" if device_id else "Audio device disconnected"
        super().__init__(msg, recoverable=True)
        self.device_id = device_id


class AudioOverflowError(AudioError):
    """Audio ring buffer overflowed — frames were dropped."""

    def __init__(self, dropped_frames: int = 0):
        super().__init__(
            f"Audio buffer overflow: {dropped_frames} frames dropped",
            recoverable=True,
        )
        self.dropped_frames = dropped_frames


class SampleRateError(AudioError):
    """The requested sample rate is not supported by the device."""

    def __init__(self, requested: int, supported: list[int] | None = None):
        supported_str = str(supported) if supported else "unknown"
        super().__init__(
            f"Sample rate {requested}Hz not supported (available: {supported_str})",
            recoverable=True,
        )
        self.requested = requested
        self.supported = supported or []


class DSPPipelineError(AudioError):
    """A DSP pipeline stage failed during processing."""

    def __init__(self, stage_name: str, cause: Exception | None = None):
        msg = f"DSP stage '{stage_name}' failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.stage_name = stage_name
        self.cause = cause


# ====================================================================== #
#  VAD / Wake Word errors
# ====================================================================== #

class VADError(SpeechError):
    """Voice Activity Detection error."""
    pass


class VADModelLoadError(VADError):
    """Failed to load VAD model."""

    def __init__(self, model_name: str = "silero_vad", cause: Exception | None = None):
        msg = f"Failed to load VAD model '{model_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.model_name = model_name


class WakeWordError(SpeechError):
    """Wake word detection error."""
    pass


class WakeWordModelLoadError(WakeWordError):
    """Failed to load wake word model."""

    def __init__(self, model_name: str = "openwakeword", cause: Exception | None = None):
        msg = f"Failed to load wake word model '{model_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.model_name = model_name


# ====================================================================== #
#  STT errors
# ====================================================================== #

class STTError(SpeechError):
    """Base for speech-to-text errors."""
    pass


class STTModelLoadError(STTError):
    """Failed to load an STT model."""

    def __init__(self, model_name: str, cause: Exception | None = None):
        msg = f"Failed to load STT model '{model_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.model_name = model_name


class STTTranscriptionError(STTError):
    """Transcription failed for a given audio segment."""

    def __init__(self, engine_name: str = "", cause: Exception | None = None):
        msg = f"Transcription failed"
        if engine_name:
            msg += f" (engine: {engine_name})"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.engine_name = engine_name


class STTTimeoutError(STTError):
    """Transcription timed out."""

    def __init__(self, timeout_s: float = 0):
        super().__init__(
            f"STT transcription timed out after {timeout_s:.1f}s",
            recoverable=True,
        )
        self.timeout_s = timeout_s


class LowConfidenceError(STTError):
    """Transcript confidence is below the action-specific threshold.

    Not a system error — this signals that clarification is needed.
    """

    def __init__(self, confidence: float, threshold: float, action: str = ""):
        msg = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
        if action:
            msg += f" for action '{action}'"
        super().__init__(msg, recoverable=False)
        self.confidence = confidence
        self.threshold = threshold
        self.action = action


class AllEnginesFailedError(STTError):
    """All STT engines in the orchestrator failed."""

    def __init__(self, engine_names: list[str] | None = None):
        names = ", ".join(engine_names) if engine_names else "all"
        super().__init__(
            f"All STT engines failed: [{names}]",
            recoverable=True,
        )
        self.engine_names = engine_names or []


# ====================================================================== #
#  TTS errors
# ====================================================================== #

class TTSError(SpeechError):
    """Base for text-to-speech errors."""
    pass


class TTSModelLoadError(TTSError):
    """Failed to load a TTS model."""

    def __init__(self, model_name: str, cause: Exception | None = None):
        msg = f"Failed to load TTS model '{model_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)
        self.model_name = model_name


class TTSSynthesisError(TTSError):
    """TTS synthesis failed for a given text."""

    def __init__(self, text_preview: str = "", cause: Exception | None = None):
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        msg = f"TTS synthesis failed for: '{preview}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)


class PlaybackError(TTSError):
    """Audio playback failed."""

    def __init__(self, cause: Exception | None = None):
        msg = "Audio playback failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)


# ====================================================================== #
#  Speaker Verification errors
# ====================================================================== #

class SpeakerVerificationError(SpeechError):
    """Speaker verification or enrollment error."""
    pass


class SpeakerNotEnrolledError(SpeakerVerificationError):
    """No voice profile is enrolled for the target speaker."""

    def __init__(self):
        super().__init__("No speaker voice profile enrolled", recoverable=False)


class SpeakerVerificationFailedError(SpeakerVerificationError):
    """Speaker did not match the enrolled voice profile."""

    def __init__(self, similarity: float = 0.0, threshold: float = 0.0):
        super().__init__(
            f"Speaker verification failed (similarity={similarity:.2f}, "
            f"threshold={threshold:.2f})",
            recoverable=False,
        )
        self.similarity = similarity
        self.threshold = threshold


# ====================================================================== #
#  Configuration errors
# ====================================================================== #

class ConfigError(SpeechError):
    """Configuration loading or validation error."""

    def __init__(self, message: str, path: str = ""):
        full_msg = message
        if path:
            full_msg += f" (file: {path})"
        super().__init__(full_msg, recoverable=False)
        self.path = path


# ====================================================================== #
#  Safety errors
# ====================================================================== #

class SafetyError(SpeechError):
    """Safety validation error."""
    pass


class CommandBlockedError(SafetyError):
    """A command was blocked by the safety validation layer."""

    def __init__(self, action: str, reason: str = ""):
        msg = f"Command '{action}' blocked by safety validator"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=False)
        self.action = action
        self.reason = reason


class ConfirmationTimeoutError(SafetyError):
    """The user did not confirm a command within the timeout period."""

    def __init__(self, action: str, timeout_s: float = 10.0):
        super().__init__(
            f"Confirmation for '{action}' timed out after {timeout_s:.1f}s",
            recoverable=False,
        )
        self.action = action
        self.timeout_s = timeout_s


# ====================================================================== #
#  GPU / Resource errors
# ====================================================================== #

class GPUUnavailableError(SpeechError):
    """GPU is requested but not available — falling back to CPU."""

    def __init__(self, cause: Exception | None = None):
        msg = "GPU unavailable, falling back to CPU"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, recoverable=True)


class InsufficientVRAMError(SpeechError):
    """Not enough GPU VRAM to load the requested model."""

    def __init__(self, required_mb: int = 0, available_mb: int = 0):
        super().__init__(
            f"Insufficient VRAM: {required_mb}MB required, {available_mb}MB available",
            recoverable=True,
        )
        self.required_mb = required_mb
        self.available_mb = available_mb
