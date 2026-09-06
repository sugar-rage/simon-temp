"""BasePlugin ABC — the contract that all SIMON plugins must implement.

Plugins extend SIMON's capabilities without modifying core code.
They receive access to the Event Bus, Configuration, and Capability
Registry on load, and can subscribe to events, publish events, and
register custom action handlers.

Example::

    class MyPlugin(BasePlugin):
        name = "my_plugin"
        version = "1.0.0"
        description = "Example plugin"

        def on_load(self, event_bus, config, capabilities):
            self._bus = event_bus
            event_bus.subscribe("vision.world_update", self._on_world)

        def on_unload(self):
            pass

        def get_event_subscriptions(self):
            return {"vision.world_update": self._on_world}

        def get_action_handlers(self):
            return {}

        def _on_world(self, event):
            pass
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Optional


class BasePlugin(abc.ABC):
    """Abstract base class for SIMON plugins.

    Subclasses must define ``name``, ``version``, and implement
    the lifecycle methods.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique plugin identifier."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Plugin version string (semver recommended)."""

    @property
    def description(self) -> str:
        """Human-readable description."""
        return ""

    @property
    def dependencies(self) -> list[str]:
        """List of capability names this plugin requires.

        The PluginManager will check these before loading.
        """
        return []

    @abc.abstractmethod
    def on_load(
        self,
        event_bus: Any,
        config: Any,
        capabilities: Any,
    ) -> None:
        """Called when the plugin is loaded.

        Parameters
        ----------
        event_bus : EventBus
            For subscribing to and publishing events.
        config : SystemConfig
            System configuration.
        capabilities : CapabilityRegistry
            Runtime capability checks.
        """

    @abc.abstractmethod
    def on_unload(self) -> None:
        """Called when the plugin is unloaded.

        The plugin should clean up all resources, remove any
        subscriptions, and release any held state.
        """

    def get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event topic → handler mappings.

        These subscriptions are automatically registered on load
        and removed on unload by the PluginManager.

        Returns
        -------
        dict[str, callable]
            Maps event topic strings to handler functions.
        """
        return {}

    def get_action_handlers(self) -> dict[str, Callable]:
        """Return action type → handler mappings.

        These handlers are registered in the ActionRegistry on load
        and removed on unload.

        Returns
        -------
        dict[str, callable]
            Maps action type strings to handler functions.
        """
        return {}
