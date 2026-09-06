"""
Structured logging for the SIMON speech subsystem.

Uses the ``logging`` stdlib with structured key-value formatting so that
every log line is machine-parseable while remaining human-readable in the
terminal.  When ``structlog`` is available, it is used for richer
structured output; otherwise falls back gracefully.

Design decision: We wrap stdlib logging rather than requiring structlog
as a hard dependency, because the speech subsystem must start reliably
even in minimal environments.  The structured fields (latency, confidence,
engine, etc.) are always included as ``extra`` dict entries.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


# Module-level logger name prefix
_LOGGER_PREFIX = "simon.speech"

# Whether structlog is available (optional dependency)
_HAS_STRUCTLOG = False
try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    pass


class StructuredFormatter(logging.Formatter):
    """A formatter that appends structured key=value pairs to log messages.

    Standard log output::

        2026-07-30 10:41:22 [INFO] simon.speech.stt: Transcription complete | latency_ms=142 confidence=0.92 engine=faster-whisper

    This keeps logs grep-able while carrying structured data for analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base message
        base = super().format(record)

        # Append structured extras (skip standard LogRecord attributes)
        _STANDARD_ATTRS = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "exc_info", "exc_text", "message",
            "taskName",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }

        if extras:
            pairs = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            return f"{base} | {pairs}"

        return base


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a speech subsystem component.

    Args:
        name: Component name (e.g. "stt", "audio.device_manager").
              Will be prefixed with "simon.speech.".

    Returns:
        A configured ``logging.Logger`` instance.
    """
    full_name = f"{_LOGGER_PREFIX}.{name}" if name else _LOGGER_PREFIX
    return logging.getLogger(full_name)


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    use_structlog: bool = True,
) -> None:
    """Configure logging for the entire speech subsystem.

    This should be called once during ``SpeechManager.start()``.  It sets
    up a console handler (with structured formatting) and optionally a
    file handler.

    Args:
        level:         Logging level string ("DEBUG", "INFO", "WARNING", etc.).
        log_file:      Optional path to a log file for persistent logging.
        use_structlog: If True and structlog is available, use it.
    """
    root_logger = logging.getLogger(_LOGGER_PREFIX)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates on reconfiguration
    root_logger.handlers.clear()

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = StructuredFormatter(
        fmt="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate output)
    root_logger.propagate = False

    root_logger.info("Speech subsystem logging configured", extra={"level": level})


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **kwargs: Any,
) -> None:
    """Log a structured event with key-value metadata.

    Convenience function that passes kwargs as ``extra`` to the logger,
    which the ``StructuredFormatter`` will append as key=value pairs.

    Args:
        logger:  Logger instance.
        level:   Logging level (e.g. ``logging.INFO``).
        message: Human-readable message.
        **kwargs: Structured key-value pairs (e.g. latency_ms=142).

    Example::

        log_event(logger, logging.INFO, "Transcription complete",
                  latency_ms=142, confidence=0.92, engine="faster-whisper")
    """
    logger.log(level, message, extra=kwargs)
