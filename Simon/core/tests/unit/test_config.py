"""Unit tests for configuration loading — deep merge, YAML, defaults."""

from __future__ import annotations

import pytest

from core.config.loader import _deep_merge, load_config
from core.config.system_config import (
    SystemConfig,
    VisionConfig,
    CameraConfig,
    SafetyConfig,
)


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"vision": {"camera": {"width": 640, "height": 480}}}
        override = {"vision": {"camera": {"width": 1920}}}
        result = _deep_merge(base, override)
        assert result["vision"]["camera"]["width"] == 1920
        assert result["vision"]["camera"]["height"] == 480

    def test_override_adds_new_keys(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        result = _deep_merge(base, override)
        assert result["a"] == {"x": 1, "y": 2}

    def test_list_replaced_not_merged(self):
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1


class TestSystemConfigDefaults:
    def test_default_config_creation(self):
        config = SystemConfig()
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.event_bus_queue_size == 10_000

    def test_default_camera_config(self):
        config = SystemConfig()
        assert config.vision.camera.width == 640
        assert config.vision.camera.height == 480
        assert config.vision.camera.fps == 30

    def test_default_safety_config(self):
        config = SystemConfig()
        assert config.safety.enabled is True
        assert "car" in config.safety.vehicle_classes

    def test_default_navigation_config(self):
        config = SystemConfig()
        assert config.navigation.gps.provider == "serial"
        assert config.navigation.routing.primary_router == "osrm"


class TestLoadConfig:
    def test_load_with_no_files_returns_defaults(self):
        config = load_config("nonexistent1.yaml", "nonexistent2.yaml")
        assert isinstance(config, SystemConfig)
        assert config.log_level == "INFO"

    def test_load_default_yaml_files(self):
        """Loading the actual default YAML files should work."""
        config = load_config(
            "core/config/defaults/system.yaml",
            "core/config/defaults/vision.yaml",
            "core/config/defaults/navigation.yaml",
            "core/config/defaults/plugins.yaml",
        )
        assert isinstance(config, SystemConfig)
        # Values should match the YAML defaults
        assert config.vision.camera.width == 640
        assert config.navigation.gps.provider == "serial"
        assert config.safety.enabled is True

    def test_load_with_user_override(self, tmp_path):
        """User config overrides should take precedence."""
        yaml = pytest.importorskip("yaml")
        user_file = tmp_path / "user_config.yaml"
        user_file.write_text("log_level: DEBUG\ndebug: true\n")
        config = load_config(
            "core/config/defaults/system.yaml",
            user_config_path=str(user_file),
        )
        assert config.log_level == "DEBUG"
        assert config.debug is True

    def test_user_override_deep_merge(self, tmp_path):
        """User config should deep-merge into defaults."""
        yaml = pytest.importorskip("yaml")
        user_file = tmp_path / "user_config.yaml"
        user_file.write_text(
            "vision:\n  camera:\n    width: 1920\n    height: 1080\n"
        )
        config = load_config(
            "core/config/defaults/system.yaml",
            "core/config/defaults/vision.yaml",
            user_config_path=str(user_file),
        )
        assert config.vision.camera.width == 1920
        assert config.vision.camera.height == 1080
        # Other vision defaults should be preserved
        assert config.vision.camera.fps == 30
