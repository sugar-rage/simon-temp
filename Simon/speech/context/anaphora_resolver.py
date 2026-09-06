"""
Anaphora Resolver — resolves deictic and anaphoric references in commands.

Handles expressions like:
- "go **there**"    → resolves to the last mentioned location
- "do **that** again" → resolves to the last executed action
- "read **it**"     → resolves to the last referenced object (e.g. sign text)
- "**again**"       → repeats the last command

Uses the ContextStore to look up recently stored referents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from speech.context.context_store import ContextStore

logger = logging.getLogger(__name__)


# Anaphoric patterns and the context slots they resolve to.
# Order matters: more specific patterns are checked first.
_ANAPHORA_RULES: List[Tuple[re.Pattern, str, str]] = [
    # ("there", "that place") → last_location
    (re.compile(r"\b(there|that place|this place)\b", re.IGNORECASE), "last_location", "location"),
    # ("that", "this") → last_object (when not part of "that place")
    (re.compile(r"\b(that|this|it)\b", re.IGNORECASE), "last_object", "object"),
    # ("again", "repeat") → triggers repeat of last_action
    (re.compile(r"\b(again|repeat|once more)\b", re.IGNORECASE), "last_action", "action"),
    # ("him", "her", "them") → last_person
    (re.compile(r"\b(him|her|them|that person)\b", re.IGNORECASE), "last_person", "person"),
]


@dataclass
class Resolution:
    """Result of an anaphora resolution attempt.

    Attributes:
        original_text:  The input command text.
        resolved_text:  Text with anaphoric references replaced.
        resolutions:    Map of replaced term → resolved value.
        had_anaphora:   True if any anaphoric references were found.
        all_resolved:   True if all found references were successfully resolved.
    """

    original_text: str
    resolved_text: str
    resolutions: Dict[str, str]
    had_anaphora: bool
    all_resolved: bool


class AnaphoraResolver:
    """Resolves anaphoric references using the conversation context store.

    Args:
        context_store: The ContextStore to read referents from.
    """

    def __init__(self, context_store: ContextStore):
        self._store = context_store

    def resolve(self, text: str) -> Resolution:
        """Attempt to resolve anaphoric references in *text*.

        Args:
            text: The raw command text (e.g. "navigate there").

        Returns:
            A Resolution with the resolved text and metadata.
        """
        resolved = text
        resolutions: Dict[str, str] = {}
        had_anaphora = False
        all_resolved = True

        for pattern, slot_name, ref_type in _ANAPHORA_RULES:
            match = pattern.search(resolved)
            if match is None:
                continue

            had_anaphora = True
            matched_text = match.group(0)
            referent = self._store.get(slot_name)

            if referent is not None:
                resolved = pattern.sub(str(referent), resolved, count=1)
                resolutions[matched_text] = str(referent)
                logger.info(
                    f"Anaphora resolved: '{matched_text}' → '{referent}' "
                    f"(slot={slot_name})"
                )
            else:
                all_resolved = False
                logger.debug(
                    f"Anaphora unresolved: '{matched_text}' "
                    f"(slot={slot_name} not found or expired)"
                )

        return Resolution(
            original_text=text,
            resolved_text=resolved,
            resolutions=resolutions,
            had_anaphora=had_anaphora,
            all_resolved=all_resolved,
        )

    def update_context_from_command(
        self,
        action: str,
        args: Optional[Dict] = None,
    ) -> None:
        """Update context slots after a command is executed.

        This should be called after a command is successfully parsed
        so that future anaphoric references can resolve.

        Args:
            action: The action name (e.g. "navigate", "read_text").
            args:   Parsed arguments (e.g. ``{"destination": "library"}``).
        """
        self._store.set("last_action", action, source="command")

        if args:
            # Extract location-like arguments
            for key in ("destination", "location", "place", "address"):
                if key in args:
                    self._store.set("last_location", args[key], source="command")
                    break

            # Extract object-like arguments
            for key in ("object", "target", "item", "text"):
                if key in args:
                    self._store.set("last_object", args[key], source="command")
                    break

            # Extract person-like arguments
            for key in ("person", "contact", "name"):
                if key in args:
                    self._store.set("last_person", args[key], source="command")
                    break
