"""Resource Manager — GPU/CPU resource budgeting and model orchestration.

Coordinates resource allocation across subsystems to prevent OOM.
Integrates with ModelRegistry for tracking and CapabilityRegistry
for runtime feature availability.

Metrics:
- ``system.gpu_memory_mb`` — gauge of estimated GPU memory usage
- ``system.memory_mb`` — gauge of estimated CPU memory usage
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from core.resources.model_registry import ModelRegistry, ModelInfo
from core.capabilities.registry import CapabilityRegistry
from core.metrics.collector import SystemMetricsCollector

logger = logging.getLogger("simon.core.resources")


class ResourceManager:
    """Manages GPU/CPU resources and model loading decisions.

    Parameters
    ----------
    model_registry : ModelRegistry
        Tracks loaded models.
    capabilities : CapabilityRegistry
        Runtime capability status.
    gpu_budget_mb : float
        Maximum GPU memory budget in MB (0 = no GPU).
    cpu_budget_mb : float
        Maximum CPU memory budget for models in MB.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        capabilities: CapabilityRegistry,
        gpu_budget_mb: float = 0.0,
        cpu_budget_mb: float = 2048.0,
    ) -> None:
        self._registry = model_registry
        self._capabilities = capabilities
        self._gpu_budget = gpu_budget_mb
        self._cpu_budget = cpu_budget_mb
        self._metrics = SystemMetricsCollector.get()
        self._gpu_available = self._detect_gpu()

    def check_resources(self) -> dict:
        """Get a snapshot of current resource usage.

        Returns
        -------
        dict
            Resource summary with GPU and CPU memory usage.
        """
        gpu_used = self._registry.get_total_memory_mb("cuda")
        cpu_used = self._registry.get_total_memory_mb("cpu")

        # Update metrics gauges
        self._metrics.set_gauge("system.gpu_memory_mb", gpu_used)
        self._metrics.set_gauge("system.memory_mb", cpu_used)

        return {
            "gpu_available": self._gpu_available,
            "gpu_budget_mb": self._gpu_budget,
            "gpu_used_mb": gpu_used,
            "gpu_free_mb": max(0, self._gpu_budget - gpu_used),
            "cpu_budget_mb": self._cpu_budget,
            "cpu_used_mb": cpu_used,
            "cpu_free_mb": max(0, self._cpu_budget - cpu_used),
            "total_models": self._registry.model_count,
            "models": [
                {
                    "name": m.name,
                    "device": m.device,
                    "size_mb": m.size_mb,
                    "priority": m.priority,
                }
                for m in self._registry.list_models()
            ],
        }

    def request_model_load(
        self,
        name: str,
        size_mb: float,
        priority: int,
        preferred_device: str = "auto",
    ) -> bool:
        """Check if a model can be loaded, evicting lower-priority models if needed.

        Parameters
        ----------
        name : str
            Model identifier.
        size_mb : float
            Model memory footprint in MB.
        priority : int
            Model priority (lower = more important).
        preferred_device : str
            ``"cuda"``, ``"cpu"``, or ``"auto"``.

        Returns
        -------
        bool
            True if the model can be loaded (possibly after evictions).
        """
        # Determine target device
        device = self._resolve_device(preferred_device)
        budget = self._gpu_budget if device == "cuda" else self._cpu_budget
        current = self._registry.get_total_memory_mb(device)

        # Enough space already?
        if current + size_mb <= budget:
            self._registry.register(name, device, size_mb, priority)
            logger.info(
                "Model %r approved for %s (%.1f/%.1f MB used)",
                name, device, current + size_mb, budget,
            )
            return True

        # Try eviction
        evicted = 0
        while current + size_mb > budget:
            candidate = self._registry.get_eviction_candidate()
            if candidate is None:
                break
            # Don't evict higher-priority models
            if candidate.priority <= priority:
                logger.warning(
                    "Cannot evict %r (priority %d) for %r (priority %d)",
                    candidate.name, candidate.priority, name, priority,
                )
                break
            self._registry.unregister(candidate.name)
            current -= candidate.size_mb
            evicted += 1
            logger.info(
                "Evicted model %r (%.1f MB) to make room for %r",
                candidate.name, candidate.size_mb, name,
            )

        if current + size_mb <= budget:
            self._registry.register(name, device, size_mb, priority)
            logger.info(
                "Model %r approved after %d evictions", name, evicted,
            )
            return True

        logger.warning(
            "Insufficient resources for model %r (need %.1f MB, have %.1f MB free on %s)",
            name, size_mb, budget - current, device,
        )
        return False

    def get_diagnostics(self) -> dict:
        """Generate diagnostic report for health monitoring.

        Returns
        -------
        dict
            Comprehensive resource diagnostics.
        """
        resources = self.check_resources()
        resources["gpu_detected"] = self._gpu_available
        resources["process_memory_mb"] = self._get_process_memory_mb()
        return resources

    # ── Internal ─────────────────────────────────────────────────────

    def _resolve_device(self, preferred: str) -> str:
        """Resolve ``"auto"`` to the best available device."""
        if preferred == "cuda" and self._gpu_available:
            return "cuda"
        if preferred == "auto" and self._gpu_available:
            return "cuda"
        return "cpu"

    @staticmethod
    def _detect_gpu() -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _get_process_memory_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
