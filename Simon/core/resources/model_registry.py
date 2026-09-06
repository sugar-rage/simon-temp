"""Model Registry — tracks loaded ML models and their resource usage.

Maintains an inventory of all ML models currently loaded in memory,
their device placement (CPU/GPU), memory footprint, and priority.
Used by the ResourceManager for eviction decisions and by the
HealthMonitor for diagnostic reports.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("simon.core.resources")


@dataclass
class ModelInfo:
    """Information about a loaded ML model.

    Attributes
    ----------
    name : str
        Model identifier (e.g. ``"yolo_detector"``, ``"insightface"``).
    device : str
        Device placement (``"cpu"`` or ``"cuda"``).
    size_mb : float
        Estimated memory footprint in megabytes.
    priority : int
        Loading priority (lower = more important, less likely to evict).
    loaded_at : float
        Timestamp when the model was loaded.
    last_used : float
        Timestamp of the most recent inference call.
    """

    name: str
    device: str = "cpu"
    size_mb: float = 0.0
    priority: int = 5
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe registry of loaded ML models.

    Provides queries for resource management decisions: total memory,
    eviction candidates, model listing, etc.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        device: str = "cpu",
        size_mb: float = 0.0,
        priority: int = 5,
    ) -> None:
        """Register a newly loaded model.

        Parameters
        ----------
        name : str
            Model identifier.
        device : str
            ``"cpu"`` or ``"cuda"``.
        size_mb : float
            Estimated memory in MB.
        priority : int
            Importance (lower = keep longer).
        """
        info = ModelInfo(
            name=name, device=device, size_mb=size_mb, priority=priority,
        )
        with self._lock:
            self._models[name] = info
        logger.info(
            "Registered model %r on %s (%.1f MB, priority=%d)",
            name, device, size_mb, priority,
        )

    def unregister(self, name: str) -> Optional[ModelInfo]:
        """Remove a model from the registry.

        Returns
        -------
        ModelInfo or None
            The removed model info, or None if not found.
        """
        with self._lock:
            info = self._models.pop(name, None)
        if info:
            logger.info("Unregistered model %r", name)
        return info

    def touch(self, name: str) -> None:
        """Update the ``last_used`` timestamp for a model."""
        with self._lock:
            info = self._models.get(name)
            if info:
                info.last_used = time.time()

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Look up a model by name."""
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> list[ModelInfo]:
        """Return all registered models."""
        with self._lock:
            return list(self._models.values())

    def get_total_memory_mb(self, device: Optional[str] = None) -> float:
        """Total memory of all registered models, optionally filtered by device.

        Parameters
        ----------
        device : str, optional
            Filter by device (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        float
            Total estimated memory in MB.
        """
        with self._lock:
            return sum(
                m.size_mb for m in self._models.values()
                if device is None or m.device == device
            )

    def get_eviction_candidate(self) -> Optional[ModelInfo]:
        """Return the best candidate for eviction (lowest priority, least recently used).

        Returns
        -------
        ModelInfo or None
            The eviction candidate, or None if no models loaded.
        """
        with self._lock:
            models = list(self._models.values())

        if not models:
            return None

        # Sort by priority (descending = least important first),
        # then by last_used (ascending = oldest first)
        return max(models, key=lambda m: (m.priority, -m.last_used))

    @property
    def model_count(self) -> int:
        """Number of currently registered models."""
        with self._lock:
            return len(self._models)
