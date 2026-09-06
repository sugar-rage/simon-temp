"""Unified exception hierarchy for SIMON.

Every subsystem exception inherits from ``SimonError`` which carries a
``recoverable`` flag.  The recovery manager uses this flag to decide
whether to attempt automatic recovery or escalate to the user.

Hierarchy::

    SimonError
    ├── VisionError
    │   ├── CameraError
    │   │   ├── CameraNotFoundError
    │   │   ├── CameraDisconnectedError
    │   │   └── FrameCaptureError
    │   ├── DetectionError
    │   │   ├── ModelLoadError
    │   │   └── InferenceError
    │   ├── OCRError
    │   └── FaceRecognitionError
    ├── NavigationError
    │   ├── GPSError
    │   │   ├── GPSUnavailableError
    │   │   └── GPSSignalLostError
    │   ├── RoutingError
    │   │   ├── GeocodingError
    │   │   ├── NoRouteFoundError
    │   │   └── NetworkError
    │   └── GuidanceError
    ├── CoreError
    │   ├── EventBusError
    │   ├── ConfigError
    │   ├── PluginError
    │   ├── ResourceError
    │   └── StateTransitionError
    └── SpeechError (FROZEN — defined in speech/errors/)
"""

from __future__ import annotations


class SimonError(Exception):
    """Base exception for all SIMON errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    recoverable : bool
        Whether automatic recovery should be attempted.
    """

    def __init__(self, message: str = "", *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable


# ── Vision Errors ────────────────────────────────────────────────────


class VisionError(SimonError):
    """Base class for all vision subsystem errors."""


class CameraError(VisionError):
    """Base class for camera-related errors."""


class CameraNotFoundError(CameraError):
    """No camera device found at the configured index."""

    def __init__(self, device_index: int = 0) -> None:
        super().__init__(
            f"Camera device {device_index} not found",
            recoverable=True,
        )
        self.device_index = device_index


class CameraDisconnectedError(CameraError):
    """Camera was connected but lost connection."""

    def __init__(self, device_id: str = "default") -> None:
        super().__init__(
            f"Camera '{device_id}' disconnected",
            recoverable=True,
        )


class FrameCaptureError(CameraError):
    """Failed to capture a frame from the camera."""

    def __init__(self, reason: str = "unknown") -> None:
        super().__init__(
            f"Frame capture failed: {reason}",
            recoverable=True,
        )


class DetectionError(VisionError):
    """Base class for detection model errors."""


class ModelLoadError(DetectionError):
    """Failed to load an ML model."""

    def __init__(self, model_name: str, reason: str = "") -> None:
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=False)
        self.model_name = model_name


class InferenceError(DetectionError):
    """Model inference failed at runtime."""

    def __init__(self, model_name: str = "", reason: str = "") -> None:
        msg = f"Inference error"
        if model_name:
            msg += f" in '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)


class OCRError(VisionError):
    """OCR processing failed."""

    def __init__(self, message: str = "OCR processing failed") -> None:
        super().__init__(message, recoverable=True)


class FaceRecognitionError(VisionError):
    """Face recognition processing failed."""

    def __init__(self, message: str = "Face recognition failed") -> None:
        super().__init__(message, recoverable=True)


# ── Navigation Errors ────────────────────────────────────────────────


class NavigationError(SimonError):
    """Base class for all navigation subsystem errors."""


class GPSError(NavigationError):
    """Base class for GPS-related errors."""


class GPSUnavailableError(GPSError):
    """GPS device is not available."""

    def __init__(self, reason: str = "device not found") -> None:
        super().__init__(f"GPS unavailable: {reason}", recoverable=True)


class GPSSignalLostError(GPSError):
    """GPS signal was acquired but has been lost."""

    def __init__(self) -> None:
        super().__init__("GPS signal lost", recoverable=True)


class RoutingError(NavigationError):
    """Base class for routing errors."""


class GeocodingError(RoutingError):
    """Failed to geocode a destination."""

    def __init__(self, destination: str) -> None:
        super().__init__(
            f"Could not find location: '{destination}'",
            recoverable=True,
        )
        self.destination = destination


class NoRouteFoundError(RoutingError):
    """No navigable route exists between origin and destination."""

    def __init__(self, destination: str = "") -> None:
        msg = "No route found"
        if destination:
            msg += f" to '{destination}'"
        super().__init__(msg, recoverable=False)


class NetworkError(RoutingError):
    """Network request failed (OSRM, Nominatim, etc.)."""

    def __init__(self, service: str = "", reason: str = "") -> None:
        msg = "Network error"
        if service:
            msg += f" contacting {service}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, recoverable=True)


class GuidanceError(NavigationError):
    """Error in turn-by-turn guidance processing."""

    def __init__(self, message: str = "Guidance error") -> None:
        super().__init__(message, recoverable=True)


# ── Core Errors ──────────────────────────────────────────────────────


class CoreError(SimonError):
    """Base class for core system errors."""


class EventBusError(CoreError):
    """Error in event bus publish/subscribe operations."""

    def __init__(self, message: str = "Event bus error") -> None:
        super().__init__(message, recoverable=True)


class ConfigError(CoreError):
    """Configuration loading or validation error."""

    def __init__(self, message: str = "Configuration error") -> None:
        super().__init__(message, recoverable=False)


class PluginError(CoreError):
    """Plugin loading or lifecycle error."""

    def __init__(self, plugin_name: str = "", message: str = "") -> None:
        msg = "Plugin error"
        if plugin_name:
            msg += f" in '{plugin_name}'"
        if message:
            msg += f": {message}"
        super().__init__(msg, recoverable=True)


class ResourceError(CoreError):
    """Resource management error (GPU, memory, model loading)."""

    def __init__(self, message: str = "Resource error") -> None:
        super().__init__(message, recoverable=True)


class StateTransitionError(CoreError):
    """Invalid state machine transition attempted."""

    def __init__(self, from_state: str, to_state: str) -> None:
        super().__init__(
            f"Invalid state transition: {from_state} → {to_state}",
            recoverable=True,
        )
        self.from_state = from_state
        self.to_state = to_state
