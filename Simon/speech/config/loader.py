"""
YAML configuration loader with validation and defaults merging.

Loads configuration from YAML files in ``config/defaults/`` and merges
with user overrides.  Produces a validated ``SpeechConfig`` instance.

Design decision: We merge defaults → user overrides → environment variables
in that order (last wins).  This allows:
1. Sane defaults for zero-config startup
2. User customisation via ``speech.yaml`` in the project root
3. Environment-variable overrides for deployment
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from speech.config.speech_config import (
    AudioConfig,
    MonitoringConfig,
    SpeakerConfig,
    SpeechConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    WakeWordConfig,
)
from speech.errors.exceptions import ConfigError
from speech.monitoring.logger import get_logger

logger = get_logger("config.loader")

# Directory containing default YAML files
_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    Values in *override* take precedence.  Nested dicts are merged
    recursively rather than replaced wholesale.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_file(path: Path) -> dict:
    """Load a single YAML file, returning an empty dict on failure."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        logger.warning("PyYAML not installed; using defaults only")
        return {}
    except Exception as e:
        logger.error(f"Failed to load YAML file {path}: {e}")
        return {}


def _dict_to_dataclass(cls, data: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(
    user_config_path: Optional[str | Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> SpeechConfig:
    """Load and validate the full speech configuration.

    Loading order (each layer overrides the previous):
    1. Built-in defaults (from ``SpeechConfig`` dataclass defaults)
    2. Default YAML files (from ``config/defaults/``)
    3. User config file (if provided)
    4. Programmatic overrides (if provided)

    Args:
        user_config_path: Path to an optional user-provided YAML override file.
        overrides:        Dict of overrides applied last (e.g. from CLI args).

    Returns:
        A validated ``SpeechConfig`` instance.

    Raises:
        ConfigError: If a required field is invalid.
    """
    # 1. Start with empty config dict
    merged: dict[str, Any] = {}

    # 2. Load default YAML files
    default_files = [
        "speech.yaml",
        "models.yaml",
        "audio.yaml",
        "devices.yaml",
        "logging.yaml",
    ]
    for fname in default_files:
        fpath = _DEFAULTS_DIR / fname
        data = _load_yaml_file(fpath)
        merged = _deep_merge(merged, data)

    # 3. Load user config
    if user_config_path:
        user_path = Path(user_config_path)
        if user_path.exists():
            user_data = _load_yaml_file(user_path)
            merged = _deep_merge(merged, user_data)
            logger.info(f"Loaded user config from {user_path}")
        else:
            logger.warning(f"User config not found: {user_path}")

    # 4. Apply overrides
    if overrides:
        merged = _deep_merge(merged, overrides)

    # 5. Build typed config from merged dict
    try:
        config = SpeechConfig(
            audio=_dict_to_dataclass(AudioConfig, merged.get("audio", {})),
            vad=_dict_to_dataclass(VADConfig, merged.get("vad", {})),
            wakeword=_dict_to_dataclass(WakeWordConfig, merged.get("wakeword", {})),
            stt=_dict_to_dataclass(STTConfig, merged.get("stt", {})),
            tts=_dict_to_dataclass(TTSConfig, merged.get("tts", {})),
            speaker=_dict_to_dataclass(SpeakerConfig, merged.get("speaker", {})),
            monitoring=_dict_to_dataclass(MonitoringConfig, merged.get("monitoring", {})),
        )

        # Merge command_map if present
        if "command_map" in merged:
            base_map = dict(config.command_map)
            raw_map = merged["command_map"]
            if isinstance(raw_map, dict):
                for k, v in raw_map.items():
                    # Safely convert YAML booleans (e.g. True/False for yes/no) to strings
                    key_str = "yes" if k is True else ("no" if k is False else str(k))
                    if isinstance(v, list):
                        phrase_list = ["yes" if p is True else ("no" if p is False else str(p)) for p in v]
                    else:
                        phrase_list = ["yes" if v is True else ("no" if v is False else str(v))]
                    base_map[key_str] = phrase_list
            config.command_map = base_map

        logger.info(
            "Speech configuration loaded",
            extra={
                "stt_engine": config.stt.engine,
                "stt_model": config.stt.model_size,
                "vad_backend": config.vad.backend,
                "wakeword": config.wakeword.wake_word,
            },
        )
        return config

    except (AssertionError, TypeError, ValueError) as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e


def load_default_config() -> SpeechConfig:
    """Load the default configuration (no user overrides).

    Convenience function for quick startup and testing.
    """
    return load_config()
