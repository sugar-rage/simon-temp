"""YAML configuration loader with deep merge.

Loads one or more YAML files, deep-merges them (later files override
earlier ones), and returns a populated ``SystemConfig`` instance.

Mirrors the pattern from ``speech/config/loader.py`` but handles
the full system configuration tree.

Usage::

    config = load_config(
        "core/config/defaults/system.yaml",
        "core/config/defaults/vision.yaml",
        user_config_path="config_user.yaml",
    )
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

from core.config.system_config import (
    SystemConfig,
    VisionConfig,
    CameraConfig,
    DetectionConfig,
    OCRConfig,
    FaceConfig,
    DepthConfig,
    NavigationConfig,
    GPSConfig,
    RoutingConfig,
    GuidanceConfig,
    PluginConfig,
    SafetyConfig,
)

logger = logging.getLogger("simon.core.config")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``.

    - Nested dicts are merged recursively.
    - Lists in ``override`` replace lists in ``base`` (no list merge).
    - Scalar values in ``override`` replace values in ``base``.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file, returning an empty dict on failure."""
    filepath = Path(path)
    if not filepath.exists():
        logger.debug("Config file not found: %s", filepath)
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, cannot load %s", filepath)
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error("Failed to load config %s: %s", filepath, e)
        return {}


def _dict_to_dataclass(data: dict, cls: type, defaults: Any = None) -> Any:
    """Convert a flat dict to a dataclass, using defaults for missing fields.

    Only sets fields that exist in the dataclass.
    """
    if defaults is None:
        defaults = cls()

    kwargs = {}
    for field_name in cls.__dataclass_fields__:
        if field_name in data:
            kwargs[field_name] = data[field_name]
        else:
            kwargs[field_name] = getattr(defaults, field_name)
    return cls(**kwargs)


def _build_config(merged: dict) -> SystemConfig:
    """Convert a merged dict into a typed SystemConfig tree."""
    # Vision subsection
    vision_data = merged.get("vision", {})
    vision = VisionConfig(
        camera=_dict_to_dataclass(
            vision_data.get("camera", {}), CameraConfig
        ),
        detection=_dict_to_dataclass(
            vision_data.get("detection", {}), DetectionConfig
        ),
        ocr=_dict_to_dataclass(vision_data.get("ocr", {}), OCRConfig),
        face=_dict_to_dataclass(vision_data.get("face", {}), FaceConfig),
        depth=_dict_to_dataclass(vision_data.get("depth", {}), DepthConfig),
        scene_analysis_interval_frames=vision_data.get(
            "scene_analysis_interval_frames", 30
        ),
        ocr_interval_frames=vision_data.get("ocr_interval_frames", 10),
    )

    # Navigation subsection
    nav_data = merged.get("navigation", {})
    navigation = NavigationConfig(
        gps=_dict_to_dataclass(nav_data.get("gps", {}), GPSConfig),
        routing=_dict_to_dataclass(
            nav_data.get("routing", {}), RoutingConfig
        ),
        guidance=_dict_to_dataclass(
            nav_data.get("guidance", {}), GuidanceConfig
        ),
    )

    # Flat sections
    plugins = _dict_to_dataclass(merged.get("plugins", {}), PluginConfig)
    safety = _dict_to_dataclass(merged.get("safety", {}), SafetyConfig)

    return SystemConfig(
        vision=vision,
        navigation=navigation,
        plugins=plugins,
        safety=safety,
        log_level=merged.get("log_level", "INFO"),
        log_file=merged.get("log_file", "simon.log"),
        log_dir=merged.get("log_dir", "logs"),
        data_dir=merged.get("data_dir", "data"),
        debug=merged.get("debug", False),
        event_bus_rate_limit_s=merged.get("event_bus_rate_limit_s", 0.05),
        event_bus_queue_size=merged.get("event_bus_queue_size", 10_000),
        watchdog_interval_s=merged.get("watchdog_interval_s", 5.0),
        health_check_interval_s=merged.get("health_check_interval_s", 30.0),
    )


def load_config(
    *default_paths: str | Path,
    user_config_path: Optional[str | Path] = None,
) -> SystemConfig:
    """Load and merge configuration files into a SystemConfig.

    Files are merged in order: earlier files are the base, later files
    override.  The user config (if provided) is applied last.

    Parameters
    ----------
    *default_paths : str | Path
        Paths to default YAML config files.
    user_config_path : str | Path, optional
        Path to a user-specific override file.

    Returns
    -------
    SystemConfig
        Fully populated configuration object.

    If no YAML files exist or PyYAML is not installed, returns
    a ``SystemConfig`` with all defaults.
    """
    merged: dict[str, Any] = {}

    for path in default_paths:
        data = _load_yaml_file(path)
        if data:
            merged = _deep_merge(merged, data)
            logger.debug("Loaded config: %s", path)

    if user_config_path:
        user_data = _load_yaml_file(user_config_path)
        if user_data:
            merged = _deep_merge(merged, user_data)
            logger.info("Loaded user config: %s", user_config_path)

    config = _build_config(merged)
    logger.info(
        "Configuration loaded: debug=%s, log_level=%s",
        config.debug,
        config.log_level,
    )
    return config
