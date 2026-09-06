"""Action Registry — maps action type strings to handler callables.

Provides a thread-safe, extensible mapping from action identifiers
(e.g. ``"speak"``, ``"navigate"``) to the functions that execute them.

Plugins can register additional handlers via the registry.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from core.models.actions import Action, ActionResult

logger = logging.getLogger("simon.core.actions")

# Handler signature: (Action) -> ActionResult
ActionHandler = Callable[[Action], ActionResult]


class ActionRegistry:
    """Thread-safe registry mapping action types to handlers.

    Parameters
    ----------
    None — handlers are registered after construction via DI wiring.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}
        self._lock = threading.Lock()

    def register(self, action_type: str, handler: ActionHandler) -> None:
        """Register a handler for an action type.

        Parameters
        ----------
        action_type : str
            Action identifier (e.g. ``"speak"``).
        handler : callable
            Function ``(Action) -> ActionResult``.
        """
        with self._lock:
            if action_type in self._handlers:
                logger.warning(
                    "Overwriting handler for action_type=%r", action_type,
                )
            self._handlers[action_type] = handler

        logger.debug("Registered handler for action_type=%r", action_type)

    def unregister(self, action_type: str) -> bool:
        """Remove a handler.

        Returns
        -------
        bool
            True if the handler existed and was removed.
        """
        with self._lock:
            removed = self._handlers.pop(action_type, None) is not None
        if removed:
            logger.debug("Unregistered handler for action_type=%r", action_type)
        return removed

    def get_handler(self, action_type: str) -> Optional[ActionHandler]:
        """Look up a handler for an action type.

        Returns
        -------
        ActionHandler or None
        """
        with self._lock:
            return self._handlers.get(action_type)

    def has_handler(self, action_type: str) -> bool:
        """Check if a handler is registered for an action type."""
        with self._lock:
            return action_type in self._handlers

    def list_actions(self) -> list[str]:
        """Return all registered action type names."""
        with self._lock:
            return list(self._handlers.keys())
