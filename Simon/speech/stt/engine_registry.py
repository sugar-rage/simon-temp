"""
STT Engine Registry — plug-in engine discovery and management.

Provides a decorator-based registration system for STT engines,
lazy-loading, and health tracking. The EngineOrchestrator uses
the registry to discover available engines without hardcoding them.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional, Type

from speech.stt.base import BaseSTTEngine

logger = logging.getLogger(__name__)


# Global engine factory registry
_ENGINE_FACTORIES: Dict[str, Callable[..., BaseSTTEngine]] = {}


def register_engine(name: str):
    """Decorator to register an STT engine factory.

    Usage::

        @register_engine("faster-whisper")
        class FasterWhisperEngine(BaseSTTEngine):
            ...
    """
    def decorator(cls: Type[BaseSTTEngine]):
        _ENGINE_FACTORIES[name] = cls
        logger.debug(f"Registered STT engine: {name}")
        return cls
    return decorator


class EngineHealth:
    """Tracks the health of a single STT engine instance."""

    def __init__(self, engine: BaseSTTEngine):
        self.engine = engine
        self.consecutive_failures: int = 0
        self.total_transcriptions: int = 0
        self.total_failures: int = 0
        self.disabled: bool = False
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0
            self.total_transcriptions += 1

    def record_failure(self, max_consecutive: int = 5) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.total_failures += 1
            if self.consecutive_failures >= max_consecutive:
                self.disabled = True
                logger.warning(
                    f"Engine '{self.engine.name}' disabled after "
                    f"{self.consecutive_failures} consecutive failures"
                )

    def re_enable(self) -> None:
        with self._lock:
            self.disabled = False
            self.consecutive_failures = 0
            logger.info(f"Engine '{self.engine.name}' re-enabled")

    @property
    def is_available(self) -> bool:
        return not self.disabled


class EngineRegistry:
    """Manages the lifecycle and health of registered STT engines.

    Engines are instantiated lazily — fallback engines are only
    created and loaded when the primary engine's confidence drops
    below threshold.
    """

    def __init__(self):
        self._engines: Dict[str, EngineHealth] = {}
        self._priority_order: List[str] = []
        self._lock = threading.Lock()

    def add_engine(
        self, engine: BaseSTTEngine, priority: Optional[int] = None
    ) -> None:
        """Register a live engine instance.

        Args:
            engine: An instantiated (but not necessarily loaded) STT engine.
            priority: Insert position in the priority list. None = append.
        """
        with self._lock:
            self._engines[engine.name] = EngineHealth(engine)
            if engine.name not in self._priority_order:
                if priority is not None:
                    self._priority_order.insert(priority, engine.name)
                else:
                    self._priority_order.append(engine.name)

        logger.info(
            f"Added STT engine '{engine.name}' at priority "
            f"{self._priority_order.index(engine.name)}"
        )

    def remove_engine(self, name: str) -> None:
        """Remove an engine from the registry."""
        with self._lock:
            self._engines.pop(name, None)
            if name in self._priority_order:
                self._priority_order.remove(name)

    def get_primary(self) -> Optional[BaseSTTEngine]:
        """Return the highest-priority available engine."""
        with self._lock:
            for name in self._priority_order:
                health = self._engines.get(name)
                if health and health.is_available:
                    return health.engine
        return None

    def get_fallbacks(self) -> List[BaseSTTEngine]:
        """Return all available engines except the primary, in priority order."""
        primary = self.get_primary()
        with self._lock:
            result = []
            for name in self._priority_order:
                health = self._engines.get(name)
                if health and health.is_available and health.engine is not primary:
                    result.append(health.engine)
            return result

    def get_health(self, name: str) -> Optional[EngineHealth]:
        """Get the health tracker for an engine."""
        return self._engines.get(name)

    def get_all_available(self) -> List[BaseSTTEngine]:
        """Return all available engines in priority order."""
        with self._lock:
            result = []
            for name in self._priority_order:
                health = self._engines.get(name)
                if health and health.is_available:
                    result.append(health.engine)
            return result

    @property
    def engine_count(self) -> int:
        return len(self._engines)

    @property
    def available_count(self) -> int:
        return sum(1 for h in self._engines.values() if h.is_available)

    @staticmethod
    def get_registered_factories() -> Dict[str, Callable[..., BaseSTTEngine]]:
        """Return all engine classes registered via the @register_engine decorator."""
        return dict(_ENGINE_FACTORIES)
