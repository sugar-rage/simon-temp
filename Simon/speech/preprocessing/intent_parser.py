"""
Intent Parser — enhanced command parsing and intent extraction.

Takes normalised, grammar-corrected, phonetic-corrected text and
maps it to an action + arguments structure.  This replaces the simple
COMMAND_MAP lookup from the legacy ``speech_input.py`` with a more
robust multi-strategy parser:

1. Exact match against known command phrases
2. Prefix match (e.g. "navigate to X" → action=navigate, args={destination: X})
3. Fuzzy match using edit distance for typo tolerance
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Result of intent parsing.

    Attributes:
        action:     Matched action name (e.g. ``"navigate"``). Empty if no match.
        args:       Extracted arguments (e.g. ``{"destination": "library"}``).
        confidence: Parser confidence in the match [0.0, 1.0].
        method:     Parsing method used (``"exact"``, ``"prefix"``, ``"fuzzy"``).
        raw_text:   The input text that was parsed.
    """

    action: str = ""
    args: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    method: str = ""
    raw_text: str = ""

    @property
    def matched(self) -> bool:
        """True if an action was successfully parsed."""
        return bool(self.action)


# Default command patterns: (regex, action, arg_extraction_group_name)
_DEFAULT_PATTERNS: List[Tuple[re.Pattern, str, Optional[str]]] = [
    # Navigation
    (re.compile(r"^navigate\s+to\s+(.+)$"), "navigate", "destination"),
    (re.compile(r"^go\s+to\s+(.+)$"), "navigate", "destination"),
    (re.compile(r"^take\s+me\s+to\s+(.+)$"), "navigate", "destination"),
    (re.compile(r"^directions?\s+to\s+(.+)$"), "navigate", "destination"),

    # Reading/OCR
    (re.compile(r"^read\s*(?:the)?\s*text$"), "read_text", None),
    (re.compile(r"^read\s+sign$"), "read_sign", None),
    (re.compile(r"^what\s+does\s+(?:it|that|this)\s+say$"), "read_text", None),

    # Scene description
    (re.compile(r"^describe\s*(?:the)?\s*scene$"), "describe_scene", None),
    (re.compile(r"^what\s+(?:do\s+you\s+see|is\s+(?:around|in\s+front))"), "describe_scene", None),
    (re.compile(r"^what\s+is\s+(?:this|that)$"), "describe_scene", None),

    # Object detection
    (re.compile(r"^detect\s+objects?$"), "detect_objects", None),
    (re.compile(r"^find\s+(.+)$"), "find_object", "object"),

    # System
    (re.compile(r"^stop$"), "stop", None),
    (re.compile(r"^help$"), "help", None),
    (re.compile(r"^battery(?:\s+(?:level|status))?$"), "battery", None),
    (re.compile(r"^(?:what\s+)?time(?:\s+is\s+it)?$"), "time", None),
    (re.compile(r"^(?:what(?:'s|\s+is)\s+the\s+)?date(?:\s+today)?$"), "date", None),
    (re.compile(r"^volume\s+(up|down)$"), "volume", "direction"),
    (re.compile(r"^(?:set\s+)?volume\s+(\d+)$"), "volume_set", "level"),

    # Communication
    (re.compile(r"^call\s+(.+)$"), "call", "contact"),
    (re.compile(r"^emergency(?:\s+call)?$"), "emergency_call", None),
]


class IntentParser:
    """Multi-strategy intent parser.

    Args:
        extra_patterns: Additional (pattern, action, arg_name) tuples.
        fuzzy_threshold: Minimum similarity ratio for fuzzy matching.
    """

    def __init__(
        self,
        extra_patterns: Optional[List[Tuple[re.Pattern, str, Optional[str]]]] = None,
        fuzzy_threshold: float = 0.75,
    ):
        self._patterns = list(_DEFAULT_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        self._fuzzy_threshold = fuzzy_threshold

        # Build exact match lookup from patterns for fast path
        self._exact_actions: Dict[str, str] = {}
        for pattern, action, _ in self._patterns:
            # Extract exact match strings (patterns with no groups)
            source = pattern.pattern
            if source.startswith("^") and source.endswith("$") and "(" not in source:
                # Remove anchors and escapes
                clean = source[1:-1].replace("\\s+", " ").replace("\\s*", "")
                self._exact_actions[clean] = action

    def parse(self, text: str) -> ParsedIntent:
        """Parse a normalised command text into an intent.

        Strategy order:
        1. Regex pattern matching (handles arguments)
        2. Fuzzy matching against known action words

        Args:
            text: Normalised command text (lowercase, no fillers).

        Returns:
            A ``ParsedIntent`` with the matched action and arguments.
        """
        text = text.strip()

        if not text:
            return ParsedIntent(raw_text=text)

        # 1. Pattern matching
        for pattern, action, arg_name in self._patterns:
            match = pattern.match(text)
            if match:
                args = {}
                if arg_name and match.groups():
                    args[arg_name] = match.group(1).strip()

                return ParsedIntent(
                    action=action,
                    args=args,
                    confidence=1.0,
                    method="pattern",
                    raw_text=text,
                )

        # 2. Fuzzy matching (simple word overlap similarity)
        best_action, best_score = self._fuzzy_match(text)
        if best_action and best_score >= self._fuzzy_threshold:
            return ParsedIntent(
                action=best_action,
                args={},
                confidence=best_score,
                method="fuzzy",
                raw_text=text,
            )

        # No match
        logger.debug(f"No intent match for: '{text}'")
        return ParsedIntent(raw_text=text)

    def _fuzzy_match(self, text: str) -> Tuple[str, float]:
        """Simple word-overlap fuzzy matching against known action words."""
        text_words = set(text.split())
        best_action = ""
        best_score = 0.0

        # Extract unique action trigger words from patterns
        action_words: Dict[str, set] = {}
        for pattern, action, _ in self._patterns:
            source = pattern.pattern.lstrip("^").rstrip("$")
            # Extract literal words (skip regex metacharacters)
            words = set(re.findall(r"[a-z]+", source))
            if action not in action_words:
                action_words[action] = set()
            action_words[action].update(words)

        for action, keywords in action_words.items():
            if not keywords:
                continue
            overlap = len(text_words & keywords)
            score = overlap / max(len(keywords), len(text_words))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score
