"""Detection tracker — cross-frame object tracking via IoU matching.

Assigns persistent ``track_id`` values to detections across consecutive
frames using simple IoU-based association.  This is a lightweight tracker
suitable for SIMON's real-time requirements without requiring a dedicated
tracking model (like DeepSORT).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.models.detection import Detection

logger = logging.getLogger("simon.vision.detection")


@dataclass
class _TrackedObject:
    """Internal tracked object state."""

    track_id: int
    detection: Detection
    last_seen: float = field(default_factory=time.time)
    frames_tracked: int = 1
    is_new: bool = True


class DetectionTracker:
    """IoU-based cross-frame detection tracker.

    Assigns persistent ``track_id`` to detections by matching them
    against tracked objects from the previous frame using IoU.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider two detections the same object.
    max_age_s : float
        Seconds before an unmatched tracked object is discarded.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_s: float = 1.0,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_s = max_age_s
        self._tracked: dict[int, _TrackedObject] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Match new detections against tracked objects.

        Each detection in the returned list has its ``track_id`` set.
        New objects get fresh IDs.  Unmatched tracked objects are aged out.

        Parameters
        ----------
        detections : list[Detection]
            Current frame detections (without track_id).

        Returns
        -------
        list[Detection]
            Same detections with ``track_id`` assigned.
        """
        now = time.time()

        # Remove stale tracked objects
        self._tracked = {
            tid: obj
            for tid, obj in self._tracked.items()
            if now - obj.last_seen < self._max_age_s
        }

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        results: list[tuple[int, int, float]] = []

        # Compute IoU between all tracked objects and new detections
        for tid, tracked in self._tracked.items():
            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou = tracked.detection.iou(det)
                if iou >= self._iou_threshold:
                    results.append((tid, i, iou))

        # Sort by IoU descending — greedily assign best matches
        results.sort(key=lambda x: x[2], reverse=True)

        for tid, det_idx, iou in results:
            if tid in matched_track_ids or det_idx in matched_det_indices:
                continue
            matched_track_ids.add(tid)
            matched_det_indices.add(det_idx)

            # Update tracked object
            detections[det_idx].track_id = tid
            self._tracked[tid].detection = detections[det_idx]
            self._tracked[tid].last_seen = now
            self._tracked[tid].frames_tracked += 1
            self._tracked[tid].is_new = False

        # Assign new IDs to unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                track_id = self._next_id
                self._next_id += 1
                det.track_id = track_id
                self._tracked[track_id] = _TrackedObject(
                    track_id=track_id,
                    detection=det,
                    last_seen=now,
                )

        return detections

    def get_new_entries(self) -> list[int]:
        """Return track_ids of objects that appeared this frame."""
        return [
            obj.track_id
            for obj in self._tracked.values()
            if obj.is_new
        ]

    def get_lost_ids(self, max_age_s: Optional[float] = None) -> list[int]:
        """Return track_ids of objects not seen recently."""
        threshold = max_age_s or self._max_age_s
        now = time.time()
        return [
            tid
            for tid, obj in self._tracked.items()
            if now - obj.last_seen >= threshold
        ]

    @property
    def active_count(self) -> int:
        return len(self._tracked)

    def reset(self) -> None:
        """Clear all tracked objects."""
        self._tracked.clear()
        self._next_id = 1
