"""Centralized logging setup for all SIMON subsystems.

Provides ``get_logger()`` — the single logging entry point for every module —
and ``configure_logging()`` for one-time system-wide configuration.

Naming convention mirrors the speech subsystem:
    ``simon.<subsystem>.<module>``

Examples::

    logger = get_logger("core.events")       # → "simon.core.events"
    logger = get_logger("vision.camera")     # → "simon.vision.camera"
    logger = get_logger("navigation.gps")    # → "simon.navigation.gps"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_LOG_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger.

    Parameters
    ----------
    name : str
        Module path, e.g. ``"core.events"`` or ``"vision.camera"``.
        Automatically prefixed with ``"simon."``.

    Returns
    -------
    logging.Logger
    """
    qualified = f"simon.{name}" if not name.startswith("simon.") else name
    return logging.getLogger(qualified)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    console: bool = True,
    subsystem_levels: Optional[dict[str, str]] = None,
) -> None:
    """Configure system-wide logging.  Call once at startup.

    Parameters
    ----------
    level : str
        Root log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    log_file : str, optional
        Filename within ``log_dir``.  If None, no file handler.
    log_dir : str
        Directory for log files.  Created if it doesn't exist.
    console : bool
        Whether to add a stderr stream handler.
    subsystem_levels : dict, optional
        Per-subsystem level overrides, e.g.
        ``{"simon.vision": "DEBUG", "simon.navigation": "WARNING"}``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("simon")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / log_file, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Apply per-subsystem overrides
    if subsystem_levels:
        for subsystem, sub_level in subsystem_levels.items():
            qualified = (
                subsystem
                if subsystem.startswith("simon.")
                else f"simon.{subsystem}"
            )
            logging.getLogger(qualified).setLevel(
                getattr(logging, sub_level.upper(), logging.INFO)
            )

    _CONFIGURED = True
    root.info("Logging configured: level=%s, file=%s", level, log_file)
