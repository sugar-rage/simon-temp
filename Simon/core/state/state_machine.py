"""System State Machine with validated transitions.

Replaces scattered boolean flags (``self._running``, ``self._navigating``,
``shutdown_event``) with a single, centralized state machine.

See implementation_plan.md Section 3.12 for the full state diagram.

States::

    STARTING → INITIALIZING → READY ⇄ LISTENING ⇄ PROCESSING
                                  ↕         ↕          ↕
                              NAVIGATING  PAUSED    DEGRADED ⇄ ERROR
                                                       ↕
                                                  SHUTTING_DOWN → STOPPED

Thread-safety: All state reads/writes are protected by a threading.Lock.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger("simon.core.state")


class SystemState(Enum):
    """Enumeration of all valid system states."""

    STARTING = auto()
    INITIALIZING = auto()
    READY = auto()
    LISTENING = auto()
    PROCESSING = auto()
    NAVIGATING = auto()
    PAUSED = auto()
    DEGRADED = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()


# Valid transition table — defines which states can transition to which.
# Each key maps to the set of states reachable from it.
_VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.STARTING: {SystemState.INITIALIZING},
    SystemState.INITIALIZING: {
        SystemState.READY,
        SystemState.DEGRADED,
        SystemState.ERROR,
    },
    SystemState.READY: {
        SystemState.LISTENING,
        SystemState.PROCESSING,
        SystemState.NAVIGATING,
        SystemState.PAUSED,
        SystemState.DEGRADED,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.LISTENING: {
        SystemState.READY,
        SystemState.PROCESSING,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.PROCESSING: {
        SystemState.READY,
        SystemState.NAVIGATING,
        SystemState.ERROR,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.NAVIGATING: {
        SystemState.READY,
        SystemState.PROCESSING,
        SystemState.DEGRADED,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.PAUSED: {
        SystemState.READY,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.DEGRADED: {
        SystemState.READY,
        SystemState.ERROR,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.ERROR: {
        SystemState.READY,
        SystemState.DEGRADED,
        SystemState.SHUTTING_DOWN,
    },
    SystemState.SHUTTING_DOWN: {SystemState.STOPPED},
    SystemState.STOPPED: set(),  # terminal state
}

# States where the system is operational (accepting commands, processing data)
_OPERATIONAL_STATES = frozenset(
    {
        SystemState.READY,
        SystemState.LISTENING,
        SystemState.PROCESSING,
        SystemState.NAVIGATING,
    }
)

# Type alias for transition listeners
TransitionListener = Callable[[SystemState, SystemState], None]


class StateMachine:
    """Thread-safe state machine with validated transitions.

    Parameters
    ----------
    initial_state : SystemState
        Starting state.  Defaults to ``STARTING``.
    """

    def __init__(
        self, initial_state: SystemState = SystemState.STARTING
    ) -> None:
        self._state = initial_state
        self._lock = threading.Lock()
        self._listeners: list[TransitionListener] = []
        self._transition_history: list[tuple[SystemState, SystemState, float]] = []

    @property
    def state(self) -> SystemState:
        """Return the current system state."""
        with self._lock:
            return self._state

    @property
    def is_operational(self) -> bool:
        """Return True if the system is in an operational state."""
        with self._lock:
            return self._state in _OPERATIONAL_STATES

    @property
    def is_stopped(self) -> bool:
        """Return True if the system has fully stopped."""
        with self._lock:
            return self._state == SystemState.STOPPED

    def can_transition(self, new_state: SystemState) -> bool:
        """Check whether a transition to ``new_state`` is valid.

        Does **not** acquire the lock — safe to call from any thread for
        advisory checks, but the result may be stale.
        """
        with self._lock:
            return new_state in _VALID_TRANSITIONS.get(self._state, set())

    def transition(self, new_state: SystemState) -> bool:
        """Attempt a state transition.

        Returns True if the transition succeeded, False if invalid.
        Invalid transitions are logged as warnings but do **not** raise.

        Thread-safety: The transition and listener notification are atomic
        with respect to other transitions.
        """
        with self._lock:
            old_state = self._state
            valid_targets = _VALID_TRANSITIONS.get(old_state, set())

            if new_state not in valid_targets:
                logger.warning(
                    "Invalid state transition: %s → %s (valid targets: %s)",
                    old_state.name,
                    new_state.name,
                    ", ".join(s.name for s in valid_targets) or "none",
                )
                return False

            self._state = new_state
            now = time.time()
            self._transition_history.append((old_state, new_state, now))
            logger.info(
                "State transition: %s → %s",
                old_state.name,
                new_state.name,
            )

            # Notify listeners while still holding the lock to ensure
            # listeners see a consistent state
            listeners = list(self._listeners)

        # Call listeners outside the lock to prevent deadlocks
        for listener in listeners:
            try:
                listener(old_state, new_state)
            except Exception:
                logger.exception(
                    "Listener error during %s → %s transition",
                    old_state.name,
                    new_state.name,
                )

        return True

    def on_transition(self, callback: TransitionListener) -> None:
        """Register a callback invoked on every successful transition.

        The callback receives ``(old_state, new_state)`` and is called
        outside the state lock to prevent deadlocks.  Listeners must
        be lightweight and non-blocking.
        """
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: TransitionListener) -> None:
        """Remove a previously registered transition listener."""
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    @property
    def history(self) -> list[tuple[str, str, float]]:
        """Return the transition history as (from, to, timestamp) tuples."""
        with self._lock:
            return [
                (old.name, new.name, ts)
                for old, new, ts in self._transition_history
            ]
