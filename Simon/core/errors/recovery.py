"""System-wide recovery manager.

Provides automatic recovery strategies for recoverable errors. Mirrors the
pattern used in ``speech/errors/recovery.py`` but extended to cover all
SIMON subsystems.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.errors.exceptions import SimonError

logger = logging.getLogger("simon.core.recovery")


@dataclass
class RecoveryAttempt:
    """Record of a single recovery attempt."""

    error_type: str
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    message: str = ""


class SystemRecoveryManager:
    """Manages automatic recovery from recoverable errors.

    Features:
    - Exponential backoff between recovery attempts.
    - Maximum retry count per error type.
    - Recovery strategy registry (callable per error class).
    - Cooldown tracking to prevent recovery storms.

    Parameters
    ----------
    max_retries : int
        Maximum recovery attempts per error type before giving up.
    base_delay_s : float
        Initial delay between retries (doubles each attempt).
    max_delay_s : float
        Maximum delay cap.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
    ) -> None:
        self._max_retries = max_retries
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._strategies: dict[type, Callable[..., bool]] = {}
        self._attempt_counts: dict[str, int] = {}
        self._history: list[RecoveryAttempt] = []

    def register_strategy(
        self, error_type: type, strategy: Callable[..., bool]
    ) -> None:
        """Register a recovery strategy for an error type.

        Parameters
        ----------
        error_type : type
            The exception class to handle.
        strategy : callable
            A callable that takes the exception and returns ``True`` on success.
        """
        self._strategies[error_type] = strategy
        logger.debug("Registered recovery strategy for %s", error_type.__name__)

    def attempt_recovery(self, error: SimonError) -> bool:
        """Attempt to recover from an error.

        Returns True if recovery succeeded, False if exhausted or no strategy.
        """
        if not error.recoverable:
            logger.warning(
                "Error is not recoverable: %s: %s",
                type(error).__name__,
                error,
            )
            return False

        error_key = type(error).__name__
        attempt_num = self._attempt_counts.get(error_key, 0) + 1

        if attempt_num > self._max_retries:
            logger.error(
                "Max retries (%d) exhausted for %s",
                self._max_retries,
                error_key,
            )
            return False

        # Find matching strategy (check MRO for inheritance)
        strategy: Optional[Callable[..., bool]] = None
        for error_cls in type(error).__mro__:
            if error_cls in self._strategies:
                strategy = self._strategies[error_cls]
                break

        if strategy is None:
            logger.warning("No recovery strategy registered for %s", error_key)
            return False

        # Exponential backoff
        delay = min(
            self._base_delay_s * (2 ** (attempt_num - 1)),
            self._max_delay_s,
        )
        logger.info(
            "Recovery attempt %d/%d for %s (delay=%.1fs)",
            attempt_num,
            self._max_retries,
            error_key,
            delay,
        )
        time.sleep(delay)

        try:
            success = strategy(error)
        except Exception as e:
            logger.error("Recovery strategy raised: %s", e)
            success = False

        self._attempt_counts[error_key] = attempt_num if not success else 0
        self._history.append(
            RecoveryAttempt(
                error_type=error_key,
                success=success,
                message=str(error),
            )
        )

        if success:
            logger.info("Recovery succeeded for %s", error_key)
        else:
            logger.warning(
                "Recovery failed for %s (attempt %d/%d)",
                error_key,
                attempt_num,
                self._max_retries,
            )

        return success

    def reset(self, error_type: Optional[str] = None) -> None:
        """Reset retry counters.  If error_type given, reset only that type."""
        if error_type:
            self._attempt_counts.pop(error_type, None)
        else:
            self._attempt_counts.clear()

    @property
    def history(self) -> list[RecoveryAttempt]:
        """Return the list of recovery attempts."""
        return list(self._history)
