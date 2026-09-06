"""
Context Manager — orchestrates the conversational context lifecycle.

Ties together the ContextStore and AnaphoraResolver, providing a
single entry point for the ListenPipeline and SpeechManager.

Responsibilities:
- Pre-process commands through anaphora resolution
- Update context after command execution
- Periodic expiration purge
- Expose context state for logging/debugging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from speech.context.anaphora_resolver import AnaphoraResolver, Resolution
from speech.context.context_store import ContextStore

logger = logging.getLogger(__name__)


class ContextManager:
    """Unified orchestrator for conversational context.

    Args:
        default_ttl_s: Default slot TTL in seconds.
        max_slots:     Maximum number of context slots.
    """

    def __init__(
        self,
        default_ttl_s: float = 120.0,
        max_slots: int = 50,
    ):
        self._store = ContextStore(
            default_ttl_s=default_ttl_s,
            max_slots=max_slots,
        )
        self._resolver = AnaphoraResolver(self._store)

    @property
    def store(self) -> ContextStore:
        """Direct access to the underlying ContextStore."""
        return self._store

    @property
    def resolver(self) -> AnaphoraResolver:
        """Direct access to the AnaphoraResolver."""
        return self._resolver

    def resolve_command_text(self, text: str) -> Resolution:
        """Resolve anaphoric references in a command.

        This is the primary pre-processing hook called by the ListenPipeline
        after STT but before command parsing.

        Args:
            text: Raw command text from STT.

        Returns:
            A Resolution with the resolved text.
        """
        resolution = self._resolver.resolve(text)

        if resolution.had_anaphora:
            if resolution.all_resolved:
                logger.info(
                    f"Context resolution: '{resolution.original_text}' "
                    f"→ '{resolution.resolved_text}'"
                )
            else:
                logger.warning(
                    f"Partial context resolution: '{resolution.original_text}' "
                    f"→ '{resolution.resolved_text}' "
                    f"(some references unresolved)"
                )

        return resolution

    def update_after_command(
        self,
        action: str,
        args: Optional[Dict] = None,
    ) -> None:
        """Update context after a command is successfully executed.

        Args:
            action: The action name (e.g. "navigate").
            args:   Parsed arguments.
        """
        self._resolver.update_context_from_command(action, args)

    def set_context(
        self,
        name: str,
        value: Any,
        ttl_s: Optional[float] = None,
        source: str = "system",
    ) -> None:
        """Manually set a context slot (e.g. from OCR or object detection)."""
        self._store.set(name, value, ttl_s=ttl_s, source=source)

    def get_context(self, name: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._store.get(name, default)

    def purge_expired(self) -> int:
        """Remove all expired context slots."""
        return self._store.purge_expired()

    def clear(self) -> None:
        """Clear all context."""
        self._store.clear()

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of active context for debugging."""
        return self._store.all_active()
