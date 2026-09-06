"""Visual Information Classifier - categorizes OCR and visual detections.

Determines what SIMON can confidently identify for immediate TTS response
versus what should be passed post-TTS to Ollama for deeper contextual analysis.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Optional, Any


@enum.unique
class VisualPriority(str, enum.Enum):
    """Priority levels for visual information processing."""
    P0_SAFETY_CRITICAL = "SAFETY_CRITICAL"
    P1_NAVIGATION_CRITICAL = "NAVIGATION_CRITICAL"
    P2_ENVIRONMENTAL = "ENVIRONMENTAL"
    P3_CONTEXTUAL = "CONTEXTUAL"
    P4_NORMAL_OCR = "NORMAL_OCR"


@dataclass
class ClassificationResult:
    """Structured output of VisualInfoClassifier."""
    priority: VisualPriority
    category: str
    raw_text: str
    canonical_key: str
    immediate_tts_response: Optional[str] = None
    is_ambiguous: bool = False
    direction: Optional[str] = None
    distance: Optional[str] = None
    confidence: float = 1.0
    extracted_data: dict[str, Any] = field(default_factory=dict)


class VisualInfoClassifier:
    """Classifies visual information and formulates immediate spoken responses."""

    _DIST_REGEX = re.compile(
        r"([A-Za-z\s\.'-]+?)\s+(\d+(?:\.\d+)?)\s*(KM|KILOMETER|KILOMETERS|M|METERS|METRES)\b",
        re.IGNORECASE,
    )

    def classify_ocr(
        self,
        text: str,
        position: Optional[str] = None,
        distance: Optional[str] = None,
    ) -> ClassificationResult:
        """Classify a detected OCR text string."""
        clean_text = text.strip()
        text_upper = clean_text.upper()
        canonical_key = re.sub(r"[^A-Z0-9]+", "_", text_upper).strip("_")

        dir_suffix = self._format_direction_suffix(position)

        # Check for ambiguity (ellipses, fragmented punctuation)
        is_ambiguous = ("..." in text) or (".." in text) or bool(re.search(r"\.{2,}", text))
        if is_ambiguous:
            is_safety_fragment = any(
                fragment in text_upper
                for fragment in ("DIV", "CLO", "DANG", "WARN", "STOP", "NO WAY", "ENTRY")
            )
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL if is_safety_fragment else VisualPriority.P3_CONTEXTUAL,
                category="ambiguous",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=None,
                is_ambiguous=True,
                direction=position,
                distance=distance,
            )

        # ?? 1. SAFETY CRITICAL (P0) ??????????????????????????????????
        if "STOP SIGN" in text_upper or text_upper == "STOP":
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="stop_sign",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Stop sign {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "ROAD CLOSED" in text_upper or "ROAD BLOCKED" in text_upper:
            resp = f"Road closed {dir_suffix}."
            if "TAKE DIVERSION" in text_upper or "DIVERSION" in text_upper:
                resp = "Road closed ahead. Take the diversion."
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="road_closure",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=resp,
                direction=position,
                distance=distance,
            )

        if any(wk in text_upper for wk in ("CONSTRUCTION", "ROAD WORK", "WORK AHEAD")):
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="construction",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Construction {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "TAKE DIVERSION" in text_upper or text_upper == "DIVERSION":
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="diversion",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Diversion {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "NO ENTRY" in text_upper or text_upper == "NO WAY":
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="no_entry",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"No entry {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "TAKE U-TURN" in text_upper or text_upper in ("U-TURN", "U TURN"):
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="u_turn",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"U-turn {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "PEDESTRIAN CROSSING" in text_upper or text_upper == "CROSSING":
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="pedestrian_crossing",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Pedestrian crossing {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "OPEN MANHOLE" in text_upper or text_upper == "MANHOLE":
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="open_manhole",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Caution, open manhole {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if "TRAFFIC SIGNAL" in text_upper or "TRAFFIC LIGHT" in text_upper:
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="traffic_signal",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Traffic signal {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if any(wk in text_upper for wk in ("DANGER", "WARNING", "CAUTION", "HAZARD", "EMERGENCY")):
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="general_warning",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Warning {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        # ?? 2. NAVIGATION CRITICAL (P1) ??????????????????????????????
        dist_match = self._DIST_REGEX.search(clean_text)
        if dist_match:
            place = dist_match.group(1).strip().title()
            dist_val = dist_match.group(2).strip()
            unit_str = dist_match.group(3).upper()
            unit_spoken = "kilometres" if ("KM" in unit_str or "KILO" in unit_str) else "metres"
            if place:
                resp = f"{place}, {dist_val} {unit_spoken} ahead."
                return ClassificationResult(
                    priority=VisualPriority.P1_NAVIGATION_CRITICAL,
                    category="distance_board",
                    raw_text=clean_text,
                    canonical_key=canonical_key,
                    immediate_tts_response=resp,
                    direction=position,
                    distance=distance,
                    extracted_data={"place": place, "distance": f"{dist_val} {unit_spoken}"},
                )

        if any(wk in text_upper for wk in ("BUS STOP", "BUS STAND", "BUS STATION", "METRO STATION", "RAILWAY STATION")):
            return ClassificationResult(
                priority=VisualPriority.P1_NAVIGATION_CRITICAL,
                category="bus_stop",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"Bus stop {dir_suffix}.",
                direction=position,
                distance=distance,
            )

        if any(re.search(pattern, text_upper) for pattern in (
            r"\bROAD\b", r"\bSTREET\b", r"\bAVENUE\b", r"\bHIGHWAY\b",
            r"\bNH\s*\d+\b", r"\bBYPASS\b", r"\bLANE\b", r"\bSALAI\b", r"\bMARG\b"
        )):
            return ClassificationResult(
                priority=VisualPriority.P1_NAVIGATION_CRITICAL,
                category="road_name",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=f"{clean_text}, {dir_suffix}.",
                direction=position,
                distance=distance,
                extracted_data={"road_name": clean_text},
            )

        # ?? 3. ENVIRONMENTAL (P2) ????????????????????????????????????
        if any(wk in text_upper for wk in ("HOSPITAL", "POLICE", "PHARMACY", "CLINIC", "EXIT", "ENTRANCE", "RESTROOM", "TOILET")):
            return ClassificationResult(
                priority=VisualPriority.P2_ENVIRONMENTAL,
                category="facility",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=None,
                direction=position,
                distance=distance,
                extracted_data={"facility": clean_text},
            )

        # ?? 4. CONTEXTUAL / SHOPS / RESTAURANTS (P3) ?????????????????
        if any(wk in text_upper for wk in ("RESTAURANT", "HOTEL", "CAFE", "BAKERY", "SUPERMARKET", "STORE", "SHOP", "MART", "BANK", "ATM", "MEALS")):
            return ClassificationResult(
                priority=VisualPriority.P3_CONTEXTUAL,
                category="place_or_restaurant",
                raw_text=clean_text,
                canonical_key=canonical_key,
                immediate_tts_response=None,
                direction=position,
                distance=distance,
                extracted_data={"place": clean_text},
            )

        # ?? 5. NORMAL OCR (P4) ???????????????????????????????????????
        return ClassificationResult(
            priority=VisualPriority.P4_NORMAL_OCR,
            category="normal_ocr",
            raw_text=clean_text,
            canonical_key=canonical_key,
            immediate_tts_response=None,
            direction=position,
            distance=distance,
        )

    def classify_entity(self, entity_cls: str, position: Optional[str] = None, distance: Optional[str] = None, confidence: float = 1.0) -> Optional[ClassificationResult]:
        """Classify visual entities (e.g. dangerous animals, major obstacles)."""
        cls_name = (entity_cls or "").lower().strip()
        dir_suffix = self._format_direction_suffix(position)

        if cls_name in ("dog", "animal") and distance in ("near", "medium"):
            canonical_key = f"ANIMAL_{cls_name.upper()}_{position or 'AHEAD'}"
            return ClassificationResult(
                priority=VisualPriority.P0_SAFETY_CRITICAL,
                category="animal_hazard",
                raw_text=cls_name,
                canonical_key=canonical_key,
                immediate_tts_response=f"Caution, {cls_name} nearby {dir_suffix}.",
                direction=position,
                distance=distance,
                confidence=confidence,
            )

        return None

    @staticmethod
    def _format_direction_suffix(position: Optional[str]) -> str:
        """Helper to create natural directional suffix."""
        if not position:
            return "ahead"
        p = position.lower().strip()
        if p == "left":
            return "on your left"
        elif p in ("slightly_left", "slightly left"):
            return "slightly on your left"
        elif p == "right":
            return "on your right"
        elif p in ("slightly_right", "slightly right"):
            return "slightly on your right"
        return "ahead"
