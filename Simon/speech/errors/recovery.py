"""
Automatic error recovery strategies for the speech subsystem.

Provides component-level recovery actions that are triggered when an
exception with ``recoverable=True`` is caught.  Recovery actions include:
- Model reloading after GPU OOM
- Device reconnection after hot-swap
- Pipeline restart after DSP failure
- Fallback activation (e.g. energy VAD when Silero fails)

Design decision: Recovery strategies are registered per exception type
and invoked by the ``SpeechManager``'s error handler.  Each strategy
has a maximum retry count and backoff to prevent infinite loops.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from speech.errors.exceptions import (
    AudioError,
    DeviceDisconnectedError,
    DeviceNotFoundError,
    DSPPipelineError,
    GPUUnavailableError,
    InsufficientVRAMError,
    SpeechError,
    STTModelLoadError,
    STTTranscriptionError,
    TTSModelLoadError,
    VADModelLoadError,
    WakeWordModelLoadError,
)
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector

logger = get_logger("errors.recovery")
metrics = get_collector()


class RecoveryAction:
    """A single recovery action with retry and backoff logic.

    Attributes:
        name:          Human-readable name for logging.
        action:        Callable that performs the recovery.
        max_retries:   Maximum recovery attempts before giving up.
        backoff_base:  Base delay between retries (seconds).
        backoff_max:   Maximum delay between retries (seconds).
    """

    def __init__(
        self,
        name: str,
        action: Callable[[], bool],
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self.name = name
        self.action = action
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        self._attempt_count = 0
        self._last_attempt: float = 0.0
        self._lock = threading.Lock()

    def execute(self) -> bool:
        """Execute the recovery action with backoff.

        Returns:
            True if recovery succeeded, False if max retries exceeded.
        """
        with self._lock:
            if self._attempt_count >= self.max_retries:
                logger.error(
                    f"Recovery '{self.name}' exhausted ({self.max_retries} retries)"
                )
                metrics.counter("recovery.exhausted")
                return False

            # Exponential backoff
            delay = min(
                self.backoff_base * (2 ** self._attempt_count),
                self.backoff_max,
            )

            # Don't retry too quickly
            elapsed = time.monotonic() - self._last_attempt
            if elapsed < delay and self._attempt_count > 0:
                remaining = delay - elapsed
                logger.debug(f"Recovery '{self.name}' backing off {remaining:.1f}s")
                time.sleep(remaining)

            self._attempt_count += 1
            self._last_attempt = time.monotonic()

        logger.info(
            f"Recovery '{self.name}' attempt {self._attempt_count}/{self.max_retries}"
        )
        metrics.counter("recovery.attempts")

        try:
            success = self.action()
            if success:
                logger.info(f"Recovery '{self.name}' succeeded")
                metrics.counter("recovery.successes")
                self.reset()
                return True
            else:
                logger.warning(f"Recovery '{self.name}' returned False")
                return False
        except Exception as e:
            logger.error(f"Recovery '{self.name}' raised: {e}")
            metrics.counter("recovery.errors")
            return False

    def reset(self) -> None:
        """Reset retry counter (after successful recovery)."""
        with self._lock:
            self._attempt_count = 0

    @property
    def retries_remaining(self) -> int:
        return max(0, self.max_retries - self._attempt_count)


class ErrorRecoveryManager:
    """Manages recovery strategies for all speech subsystem components.

    Components register recovery actions for specific exception types.
    When an error occurs, the manager looks up and executes the
    appropriate recovery action.

    Usage::

        recovery = ErrorRecoveryManager()
        recovery.register(DeviceDisconnectedError, RecoveryAction(
            name="reconnect_audio",
            action=lambda: device_manager.get_best_device() is not None,
        ))

        # Later, when an error occurs:
        if recovery.attempt_recovery(error):
            # Recovery succeeded, continue operation
        else:
            # Recovery failed, escalate
    """

    def __init__(self):
        self._strategies: dict[type, RecoveryAction] = {}
        self._lock = threading.Lock()

    def register(
        self,
        exception_type: type,
        action: RecoveryAction,
    ) -> None:
        """Register a recovery action for an exception type.

        Args:
            exception_type: The exception class to handle.
            action:         The recovery action to execute.
        """
        with self._lock:
            self._strategies[exception_type] = action
        logger.debug(f"Registered recovery '{action.name}' for {exception_type.__name__}")

    def attempt_recovery(self, error: Exception) -> bool:
        """Attempt recovery for the given error.

        Looks up the recovery action by exception type (checking
        parent classes for inheritance-based matching).

        Args:
            error: The exception that triggered recovery.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        # Check if the error is recoverable
        if isinstance(error, SpeechError) and not error.recoverable:
            logger.info(f"Error is non-recoverable: {type(error).__name__}")
            return False

        # Find matching strategy (check exact type first, then parents)
        action = None
        with self._lock:
            error_type = type(error)
            if error_type in self._strategies:
                action = self._strategies[error_type]
            else:
                # Check parent classes
                for exc_type, strategy in self._strategies.items():
                    if isinstance(error, exc_type):
                        action = strategy
                        break

        if action is None:
            logger.warning(f"No recovery strategy for {type(error).__name__}")
            return False

        logger.info(
            f"Attempting recovery '{action.name}' for {type(error).__name__}",
            extra={"error": str(error)[:100]},
        )
        return action.execute()

    def reset_all(self) -> None:
        """Reset retry counters for all strategies."""
        with self._lock:
            for action in self._strategies.values():
                action.reset()
