"""Visual Context package for SIMON."""
from core.visual_context.classifier import VisualInfoClassifier, VisualPriority, ClassificationResult
from core.visual_context.event_tracker import VisualEventTracker
from core.visual_context.ollama_client import OllamaClient
from core.visual_context.context_memory import ContextualVisualMemory, VisualMemoryEntry
from core.visual_context.visual_information_manager import VisualInformationManager, VisualEvent

__all__ = [
    "VisualInfoClassifier",
    "VisualPriority",
    "ClassificationResult",
    "VisualEventTracker",
    "OllamaClient",
    "ContextualVisualMemory",
    "VisualMemoryEntry",
    "VisualInformationManager",
    "VisualEvent",
]
