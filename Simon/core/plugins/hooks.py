"""Hook system — thread-safe hook point registration and invocation.

Hook points are well-defined extension points in the SIMON pipeline
where plugins can inject custom logic.  The HookRegistry manages
callback registration and invocation per hook point.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger("simon.core.plugins")


class HookPoint(Enum):
    """Well-defined extension points in the SIMON pipeline."""

    PRE_DETECTION = auto()
    """Before object detection runs on a frame."""

    POST_DETECTION = auto()
    """After object detection, with raw detections."""

    PRE_ANNOUNCE = auto()
    """Before an announcement is spoken (can modify text)."""

    POST_ANNOUNCE = auto()
    """After an announcement is spoken."""

    ON_COMMAND = auto()
    """When a voice command is received (before planning)."""

    ON_STARTUP = auto()
    """During system startup, after subsystems initialized."""

    ON_SHUTDOWN = auto()
    """During system shutdown, before subsystems stop."""


# Callback signature
HookCallback = Callable[..., Optional[Any]]


class HookRegistry:
    """Thread-safe registry for hook callbacks.

    Plugins register callbacks at specific hook points.  When a hook
    is invoked, all registered callbacks run sequentially in
    registration order.

    Parameters
    ----------
    None — construct directly.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[str, HookCallback]]] = {
            point: [] for point in HookPoint
        }
        self._lock = threading.Lock()

    def register(
        self, hook_point: HookPoint, callback: HookCallback, plugin_name: str,
    ) -> None:
        """Register a callback at a hook point.

        Parameters
        ----------
        hook_point : HookPoint
            Where to hook in.
        callback : callable
            The function to call.
        plugin_name : str
            Owning plugin name (for targeted unregistration).
        """
        with self._lock:
            self._hooks[hook_point].append((plugin_name, callback))
        logger.debug(
            "Registered hook %s for plugin %r", hook_point.name, plugin_name,
        )

    def unregister_all(self, plugin_name: str) -> int:
        """Remove all hooks registered by a specific plugin.

        Parameters
        ----------
        plugin_name : str
            Plugin whose hooks to remove.

        Returns
        -------
        int
            Number of hooks removed.
        """
        removed = 0
        with self._lock:
            for point in HookPoint:
                before = len(self._hooks[point])
                self._hooks[point] = [
                    (name, cb) for name, cb in self._hooks[point]
                    if name != plugin_name
                ]
                removed += before - len(self._hooks[point])
        if removed:
            logger.debug("Unregistered %d hooks for plugin %r", removed, plugin_name)
        return removed

    def invoke(self, hook_point: HookPoint, **kwargs: Any) -> list[Any]:
        """Invoke all callbacks registered at a hook point.

        Parameters
        ----------
        hook_point : HookPoint
            Which hook to fire.
        **kwargs
            Arguments passed to each callback.

        Returns
        -------
        list[Any]
            Return values from each callback (None values excluded).
        """
        with self._lock:
            callbacks = list(self._hooks[hook_point])

        results = []
        for plugin_name, callback in callbacks:
            try:
                result = callback(**kwargs)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.error(
                    "Hook %s callback from plugin %r failed",
                    hook_point.name, plugin_name,
                    exc_info=True,
                )
        return results

    def get_hook_count(self, hook_point: Optional[HookPoint] = None) -> int:
        """Return the number of registered hooks.

        Parameters
        ----------
        hook_point : HookPoint, optional
            Count for a specific point.  If None, count all.
        """
        with self._lock:
            if hook_point is not None:
                return len(self._hooks[hook_point])
            return sum(len(cbs) for cbs in self._hooks.values())
