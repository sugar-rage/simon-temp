"""
Confirmation Manager — voice confirmation dialog state machine.

When the SafetyValidator determines that a command needs confirmation
(e.g. "Navigate to highway"), the ConfirmationManager:
1. Speaks a confirmation prompt via the SpeakPipeline
2. Listens for a yes/no response
3. Returns the confirmation decision

The state machine supports timeouts and cancellation.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ConfirmationState(Enum):
    """States of the confirmation dialog."""

    IDLE = auto()
    """No confirmation in progress."""

    WAITING = auto()
    """Waiting for user's yes/no response."""

    CONFIRMED = auto()
    """User confirmed the action."""

    DENIED = auto()
    """User denied the action."""

    TIMED_OUT = auto()
    """Confirmation timed out without response."""

    CANCELLED = auto()
    """Confirmation was cancelled programmatically."""


# Patterns that indicate confirmation
_YES_PATTERNS = frozenset({
    "yes", "yeah", "yep", "yup", "correct", "confirm",
    "affirmative", "sure", "ok", "okay", "go ahead",
    "do it", "proceed", "right", "that's right", "absolutely",
})

# Patterns that indicate denial
_NO_PATTERNS = frozenset({
    "no", "nope", "nah", "cancel", "stop", "don't",
    "negative", "abort", "wrong", "not that", "never mind",
    "nevermind", "forget it",
})


class ConfirmationManager:
    """Manages voice confirmation dialogs.

    Args:
        timeout_s:    Seconds to wait for a response before timing out.
        speak_fn:     Callback to speak a prompt (e.g. ``SpeechManager.speak``).
                       Signature: ``speak_fn(text: str) -> None``.
    """

    def __init__(
        self,
        timeout_s: float = 10.0,
        speak_fn: Optional[Callable[[str], None]] = None,
    ):
        self._timeout_s = timeout_s
        self._speak_fn = speak_fn
        self._state = ConfirmationState.IDLE
        self._pending_action: Optional[str] = None
        self._pending_message: Optional[str] = None
        self._request_time: float = 0.0

    @property
    def state(self) -> ConfirmationState:
        """Current confirmation state."""
        # Auto-timeout check
        if self._state == ConfirmationState.WAITING:
            if (time.monotonic() - self._request_time) > self._timeout_s:
                self._state = ConfirmationState.TIMED_OUT
                logger.info(
                    f"Confirmation timed out for action '{self._pending_action}'"
                )
        return self._state

    @property
    def is_waiting(self) -> bool:
        """True if waiting for user confirmation."""
        return self.state == ConfirmationState.WAITING

    @property
    def pending_action(self) -> Optional[str]:
        """The action awaiting confirmation."""
        return self._pending_action

    def request_confirmation(self, action: str, message: str) -> None:
        """Start a confirmation dialog.

        Args:
            action:  Action name that requires confirmation.
            message: Confirmation prompt to speak to the user.
        """
        self._state = ConfirmationState.WAITING
        self._pending_action = action
        self._pending_message = message
        self._request_time = time.monotonic()

        logger.info(f"Requesting confirmation for '{action}': {message}")

        if self._speak_fn:
            self._speak_fn(message)

    def process_response(self, text: str) -> ConfirmationState:
        """Process a user response during a confirmation dialog.

        Args:
            text: The recognized speech text.

        Returns:
            The new confirmation state.
        """
        if self._state != ConfirmationState.WAITING:
            return self._state

        normalized = text.strip().lower()

        # Check for yes
        if any(p in normalized for p in _YES_PATTERNS):
            self._state = ConfirmationState.CONFIRMED
            logger.info(
                f"Confirmation CONFIRMED for '{self._pending_action}' "
                f"(response: '{text}')"
            )
            return self._state

        # Check for no
        if any(p in normalized for p in _NO_PATTERNS):
            self._state = ConfirmationState.DENIED
            logger.info(
                f"Confirmation DENIED for '{self._pending_action}' "
                f"(response: '{text}')"
            )
            return self._state

        # Ambiguous response — re-prompt
        logger.debug(
            f"Ambiguous confirmation response: '{text}', staying in WAITING"
        )
        if self._speak_fn:
            self._speak_fn("Please say yes or no.")

        return self._state

    def cancel(self) -> None:
        """Cancel the current confirmation dialog."""
        if self._state == ConfirmationState.WAITING:
            self._state = ConfirmationState.CANCELLED
            logger.info(f"Confirmation cancelled for '{self._pending_action}'")

    def reset(self) -> None:
        """Reset to IDLE state."""
        self._state = ConfirmationState.IDLE
        self._pending_action = None
        self._pending_message = None
        self._request_time = 0.0
