"""Plugin Manager — discovery, validation, loading, and lifecycle management.

Scans a plugin directory for plugin modules, validates their dependencies
against the CapabilityRegistry, loads them via ``on_load()``, wires their
event subscriptions and action handlers, and manages graceful unloading.

Plugin discovery:
- Scans ``plugins/`` directory for Python packages
- Each package must contain a class that extends ``BasePlugin``
- Optionally reads ``plugin.yaml`` for metadata and dependency declaration

Events published:
- ``plugin.loaded`` — plugin successfully initialized
- ``plugin.unloaded`` — plugin cleanly unloaded
- ``plugin.error`` — plugin encountered an error during lifecycle
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from typing import Any, Optional

from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.capabilities.registry import CapabilityRegistry
from core.plugins.plugin_base import BasePlugin
from core.plugins.hooks import HookRegistry
from core.actions.action_registry import ActionRegistry

logger = logging.getLogger("simon.core.plugins")


class PluginManager:
    """Manages the lifecycle of all SIMON plugins.

    Parameters
    ----------
    event_bus : EventBus
        For plugin event subscriptions and lifecycle events.
    config : Any
        System configuration (passed to plugins on load).
    capabilities : CapabilityRegistry
        For dependency checking.
    action_registry : ActionRegistry
        For registering plugin action handlers.
    hook_registry : HookRegistry
        For managing plugin hooks.
    plugin_dir : str
        Directory to scan for plugins.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Any,
        capabilities: CapabilityRegistry,
        action_registry: ActionRegistry,
        hook_registry: HookRegistry,
        plugin_dir: str = "plugins",
    ) -> None:
        self._event_bus = event_bus
        self._config = config
        self._capabilities = capabilities
        self._action_registry = action_registry
        self._hook_registry = hook_registry
        self._plugin_dir = plugin_dir
        self._plugins: dict[str, BasePlugin] = {}
        self._lock = threading.Lock()

    def discover_plugins(self) -> list[str]:
        """Scan the plugin directory for available plugins.

        Returns
        -------
        list[str]
            Names of discovered plugin directories.
        """
        if not os.path.isdir(self._plugin_dir):
            logger.info("Plugin directory %r does not exist", self._plugin_dir)
            return []

        discovered = []
        for entry in os.listdir(self._plugin_dir):
            entry_path = os.path.join(self._plugin_dir, entry)
            init_path = os.path.join(entry_path, "__init__.py")
            if os.path.isdir(entry_path) and os.path.isfile(init_path):
                discovered.append(entry)

        logger.info("Discovered %d plugins: %s", len(discovered), discovered)
        return discovered

    def load_plugin(self, name: str) -> bool:
        """Load a single plugin by name.

        Parameters
        ----------
        name : str
            Plugin directory name.

        Returns
        -------
        bool
            True if the plugin loaded successfully.
        """
        with self._lock:
            if name in self._plugins:
                logger.warning("Plugin %r already loaded", name)
                return True

        try:
            # Import the plugin module
            module_path = f"{self._plugin_dir}.{name}"
            module = importlib.import_module(module_path)

            # Find the BasePlugin subclass
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BasePlugin)
                    and attr is not BasePlugin
                ):
                    plugin_class = attr
                    break

            if plugin_class is None:
                logger.error("No BasePlugin subclass found in %r", module_path)
                self._publish_error(name, "No BasePlugin subclass found")
                return False

            # Instantiate
            plugin = plugin_class()

            # Check dependencies
            missing = [
                dep for dep in plugin.dependencies
                if not self._capabilities.is_available(dep)
            ]
            if missing:
                logger.warning(
                    "Plugin %r has unmet dependencies: %s", name, missing,
                )
                self._publish_error(name, f"Missing dependencies: {missing}")
                return False

            # Load
            plugin.on_load(self._event_bus, self._config, self._capabilities)

            # Wire event subscriptions
            for topic, handler in plugin.get_event_subscriptions().items():
                self._event_bus.subscribe(topic, handler, source=f"plugin.{name}")

            # Wire action handlers
            for action_type, handler in plugin.get_action_handlers().items():
                self._action_registry.register(action_type, handler)

            with self._lock:
                self._plugins[name] = plugin

            logger.info("Loaded plugin %r v%s", plugin.name, plugin.version)
            self._publish_loaded(plugin)
            return True

        except Exception as exc:
            logger.error("Failed to load plugin %r: %s", name, exc, exc_info=True)
            self._publish_error(name, str(exc))
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name.

        Parameters
        ----------
        name : str
            Plugin name.

        Returns
        -------
        bool
            True if the plugin was found and unloaded.
        """
        with self._lock:
            plugin = self._plugins.pop(name, None)

        if plugin is None:
            return False

        try:
            # Remove event subscriptions
            for topic, handler in plugin.get_event_subscriptions().items():
                self._event_bus.unsubscribe(handler)

            # Remove action handlers
            for action_type in plugin.get_action_handlers():
                self._action_registry.unregister(action_type)

            # Remove hooks
            self._hook_registry.unregister_all(name)

            # Unload
            plugin.on_unload()

            logger.info("Unloaded plugin %r", name)
            self._publish_unloaded(name)
            return True

        except Exception as exc:
            logger.error("Error unloading plugin %r: %s", name, exc, exc_info=True)
            self._publish_error(name, f"Unload error: {exc}")
            return False

    def load_all(self) -> int:
        """Discover and load all available plugins.

        Returns
        -------
        int
            Number of plugins successfully loaded.
        """
        discovered = self.discover_plugins()
        loaded = 0
        for name in discovered:
            if self.load_plugin(name):
                loaded += 1
        logger.info("Loaded %d/%d discovered plugins", loaded, len(discovered))
        return loaded

    def unload_all(self) -> int:
        """Unload all loaded plugins.

        Returns
        -------
        int
            Number of plugins unloaded.
        """
        with self._lock:
            names = list(self._plugins.keys())

        unloaded = 0
        for name in names:
            if self.unload_plugin(name):
                unloaded += 1
        return unloaded

    def get_loaded_plugins(self) -> list[str]:
        """Return names of all currently loaded plugins."""
        with self._lock:
            return list(self._plugins.keys())

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin instance by name."""
        with self._lock:
            return self._plugins.get(name)

    # ── Event publishing ─────────────────────────────────────────────

    def _publish_loaded(self, plugin: BasePlugin) -> None:
        try:
            self._event_bus.publish(Event(
                topic=event_types.PLUGIN_LOADED,
                priority=Priority.LOW,
                source="plugin_manager",
                data={"name": plugin.name, "version": plugin.version},
            ))
        except Exception:
            logger.error("Failed to publish plugin.loaded", exc_info=True)

    def _publish_unloaded(self, name: str) -> None:
        try:
            self._event_bus.publish(Event(
                topic=event_types.PLUGIN_UNLOADED,
                priority=Priority.LOW,
                source="plugin_manager",
                data={"name": name},
            ))
        except Exception:
            logger.error("Failed to publish plugin.unloaded", exc_info=True)

    def _publish_error(self, name: str, error: str) -> None:
        try:
            self._event_bus.publish(Event(
                topic=event_types.PLUGIN_ERROR,
                priority=Priority.INFORMATIONAL,
                source="plugin_manager",
                data={"name": name, "error": error},
            ))
        except Exception:
            logger.error("Failed to publish plugin.error", exc_info=True)
