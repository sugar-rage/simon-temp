"""Unit tests for PluginManager, BasePlugin, and HookRegistry."""

from __future__ import annotations

import pytest

from core.events.event_bus import EventBus
from core.capabilities.registry import CapabilityRegistry
from core.actions.action_registry import ActionRegistry
from core.models.actions import Action, ActionResult
from core.plugins.plugin_base import BasePlugin
from core.plugins.hooks import HookPoint, HookRegistry
from core.plugins.plugin_manager import PluginManager


# ── Test fixtures ────────────────────────────────────────────────────


class MockPlugin(BasePlugin):
    """A test plugin for unit testing."""

    _name = "mock_plugin"
    _version = "1.0.0"
    _loaded = False
    _unloaded = False
    _events_received = []

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def description(self):
        return "A mock plugin for testing"

    def on_load(self, event_bus, config, capabilities):
        self._loaded = True
        self._bus = event_bus

    def on_unload(self):
        self._unloaded = True

    def get_event_subscriptions(self):
        return {"test.event": self._on_event}

    def get_action_handlers(self):
        return {"mock_action": self._handle_action}

    def _on_event(self, event):
        self._events_received.append(event)

    def _handle_action(self, action):
        return ActionResult(action=action, success=True, message="mock handled")


class PluginWithDeps(BasePlugin):
    """Plugin requiring unavailable capability."""

    @property
    def name(self):
        return "deps_plugin"

    @property
    def version(self):
        return "1.0.0"

    @property
    def dependencies(self):
        return ["capability.quantum_computer"]

    def on_load(self, event_bus, config, capabilities):
        pass

    def on_unload(self):
        pass


@pytest.fixture
def event_bus():
    bus = EventBus(rate_limit_s=0.0, enable_metrics=False)
    yield bus
    if bus.is_running:
        bus.stop()
    bus.clear()


@pytest.fixture
def capabilities():
    return CapabilityRegistry()


@pytest.fixture
def action_registry():
    return ActionRegistry()


@pytest.fixture
def hook_registry():
    return HookRegistry()


# ── HookRegistry tests ─────────────────────────────────────────────


class TestHookRegistry:
    def test_register_and_invoke(self, hook_registry):
        results = []
        hook_registry.register(
            HookPoint.ON_STARTUP, lambda: results.append("fired"), "test",
        )
        hook_registry.invoke(HookPoint.ON_STARTUP)
        assert results == ["fired"]

    def test_invoke_with_kwargs(self, hook_registry):
        results = []
        hook_registry.register(
            HookPoint.PRE_DETECTION,
            lambda frame=None: results.append(frame),
            "test",
        )
        hook_registry.invoke(HookPoint.PRE_DETECTION, frame="frame_data")
        assert results == ["frame_data"]

    def test_unregister_all_by_plugin(self, hook_registry):
        hook_registry.register(HookPoint.ON_STARTUP, lambda: None, "plugin_a")
        hook_registry.register(HookPoint.ON_SHUTDOWN, lambda: None, "plugin_a")
        hook_registry.register(HookPoint.ON_STARTUP, lambda: None, "plugin_b")

        removed = hook_registry.unregister_all("plugin_a")
        assert removed == 2
        assert hook_registry.get_hook_count(HookPoint.ON_STARTUP) == 1
        assert hook_registry.get_hook_count(HookPoint.ON_SHUTDOWN) == 0

    def test_invoke_exception_handling(self, hook_registry):
        """Callback exception should not prevent other callbacks from running."""
        results = []

        def bad_callback():
            raise RuntimeError("boom")

        def good_callback():
            results.append("ok")

        hook_registry.register(HookPoint.ON_STARTUP, bad_callback, "bad")
        hook_registry.register(HookPoint.ON_STARTUP, good_callback, "good")
        hook_registry.invoke(HookPoint.ON_STARTUP)
        assert results == ["ok"]

    def test_get_hook_count_all(self, hook_registry):
        hook_registry.register(HookPoint.ON_STARTUP, lambda: None, "a")
        hook_registry.register(HookPoint.ON_SHUTDOWN, lambda: None, "b")
        assert hook_registry.get_hook_count() == 2

    def test_invoke_returns_results(self, hook_registry):
        hook_registry.register(HookPoint.PRE_ANNOUNCE, lambda: "modified", "a")
        hook_registry.register(HookPoint.PRE_ANNOUNCE, lambda: None, "b")
        results = hook_registry.invoke(HookPoint.PRE_ANNOUNCE)
        assert results == ["modified"]  # None values excluded


# ── BasePlugin tests ───────────────────────────────────────────────


class TestBasePlugin:
    def test_mock_plugin_properties(self):
        plugin = MockPlugin()
        assert plugin.name == "mock_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.description == "A mock plugin for testing"
        assert plugin.dependencies == []

    def test_mock_plugin_lifecycle(self):
        plugin = MockPlugin()
        plugin.on_load(None, None, None)
        assert plugin._loaded
        plugin.on_unload()
        assert plugin._unloaded

    def test_mock_plugin_subscriptions(self):
        plugin = MockPlugin()
        subs = plugin.get_event_subscriptions()
        assert "test.event" in subs

    def test_mock_plugin_action_handlers(self):
        plugin = MockPlugin()
        handlers = plugin.get_action_handlers()
        assert "mock_action" in handlers


# ── PluginManager tests ─────────────────────────────────────────────


class TestPluginManager:
    def test_discover_nonexistent_dir(
        self, event_bus, capabilities, action_registry, hook_registry,
    ):
        mgr = PluginManager(
            event_bus=event_bus,
            config=None,
            capabilities=capabilities,
            action_registry=action_registry,
            hook_registry=hook_registry,
            plugin_dir="nonexistent_plugin_dir_xyz",
        )
        discovered = mgr.discover_plugins()
        assert discovered == []

    def test_get_loaded_plugins_empty(
        self, event_bus, capabilities, action_registry, hook_registry,
    ):
        mgr = PluginManager(
            event_bus=event_bus,
            config=None,
            capabilities=capabilities,
            action_registry=action_registry,
            hook_registry=hook_registry,
        )
        assert mgr.get_loaded_plugins() == []

    def test_load_nonexistent_plugin(
        self, event_bus, capabilities, action_registry, hook_registry,
    ):
        mgr = PluginManager(
            event_bus=event_bus,
            config=None,
            capabilities=capabilities,
            action_registry=action_registry,
            hook_registry=hook_registry,
        )
        result = mgr.load_plugin("nonexistent_plugin_xyz")
        assert result is False
