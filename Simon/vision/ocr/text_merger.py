"""OCR text merger — deduplicates and consolidates OCR results.

Handles merging overlapping text boxes and cooldown tracking to avoid
re-announcing the same text repeatedly.
"""

from __future__ import annotations

import time
from typing import Optional

from vision.ocr.base import OCRResult


class TextMerger:
    """Merges overlapping OCR results and tracks cooldowns.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to merge two text boxes.
    cooldown_s : float
        Seconds before the same text can be announced again.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        cooldown_s: float = 10.0,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._cooldown_s = cooldown_s
        self._last_announced: dict[str, float] = {}

    def merge(self, results: list[OCRResult]) -> list[OCRResult]:
        """Merge overlapping text regions and filter by cooldown.

        Returns the merged, non-duplicate results.
        """
        if not results:
            return []

        # Sort by confidence descending
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        merged: list[OCRResult] = []
        used: set[int] = set()

        for i, primary in enumerate(sorted_results):
            if i in used:
                continue

            combined_text = primary.text
            best_confidence = primary.confidence
            best_bbox = primary.bbox

            for j, candidate in enumerate(sorted_results):
                if j <= i or j in used:
                    continue
                if (
                    best_bbox is not None
                    and candidate.bbox is not None
                    and self._compute_iou(best_bbox, candidate.bbox)
                    >= self._iou_threshold
                ):
                    # Merge text
                    combined_text += " " + candidate.text
                    used.add(j)

            merged.append(
                OCRResult(
                    text=combined_text.strip(),
                    confidence=best_confidence,
                    bbox=best_bbox,
                )
            )
            used.add(i)

        # Apply cooldown filter
        now = time.time()
        filtered: list[OCRResult] = []
        for result in merged:
            key = result.text.lower().strip()
            last = self._last_announced.get(key, 0.0)
            if now - last >= self._cooldown_s:
                filtered.append(result)
                self._last_announced[key] = now

        return filtered

    def _compute_iou(
        self,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        area_a = max(0, bbox_a[2] - bbox_a[0]) * max(0, bbox_a[3] - bbox_a[1])
        area_b = max(0, bbox_b[2] - bbox_b[0]) * max(0, bbox_b[3] - bbox_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def reset_cooldowns(self) -> None:
        """Clear cooldown tracking."""
        self._last_announced.clear()
