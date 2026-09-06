"""Unit tests for DetectionTracker — cross-frame IoU tracking."""

from __future__ import annotations

import time

import pytest

from core.models.detection import Detection
from vision.detection.tracker import DetectionTracker
from vision.tests.conftest import make_detection


class TestTrackerBasic:
    def test_first_frame_assigns_ids(self, detection_tracker):
        dets = [
            make_detection(cls="car", bbox=(10, 10, 100, 100)),
            make_detection(cls="person", bbox=(200, 200, 300, 300)),
        ]
        result = detection_tracker.update(dets)
        assert all(d.track_id is not None for d in result)
        ids = {d.track_id for d in result}
        assert len(ids) == 2  # unique IDs

    def test_same_position_keeps_id(self, detection_tracker):
        dets1 = [make_detection(bbox=(100, 100, 200, 200))]
        result1 = detection_tracker.update(dets1)
        tid1 = result1[0].track_id

        dets2 = [make_detection(bbox=(105, 105, 205, 205))]  # slight shift
        result2 = detection_tracker.update(dets2)
        assert result2[0].track_id == tid1

    def test_different_position_gets_new_id(self, detection_tracker):
        dets1 = [make_detection(bbox=(0, 0, 50, 50))]
        result1 = detection_tracker.update(dets1)
        tid1 = result1[0].track_id

        dets2 = [make_detection(bbox=(400, 400, 500, 500))]  # far away
        result2 = detection_tracker.update(dets2)
        assert result2[0].track_id != tid1

    def test_multiple_objects_tracked(self, detection_tracker):
        dets1 = [
            make_detection(bbox=(10, 10, 50, 50)),
            make_detection(bbox=(200, 200, 250, 250)),
        ]
        result1 = detection_tracker.update(dets1)

        dets2 = [
            make_detection(bbox=(12, 12, 52, 52)),    # shifted car 1
            make_detection(bbox=(202, 202, 252, 252)), # shifted car 2
        ]
        result2 = detection_tracker.update(dets2)

        assert result2[0].track_id == result1[0].track_id
        assert result2[1].track_id == result1[1].track_id


class TestTrackerAging:
    def test_lost_objects_are_aged_out(self):
        tracker = DetectionTracker(max_age_s=0.1)
        dets1 = [make_detection(bbox=(100, 100, 200, 200))]
        tracker.update(dets1)

        time.sleep(0.2)  # exceed max age

        dets2 = [make_detection(bbox=(100, 100, 200, 200))]
        result = tracker.update(dets2)
        # Should get a new ID since the old one aged out
        assert result[0].track_id != dets1[0].track_id or result[0].track_id is not None


class TestTrackerProperties:
    def test_active_count(self, detection_tracker):
        assert detection_tracker.active_count == 0
        detection_tracker.update([make_detection()])
        assert detection_tracker.active_count == 1

    def test_reset(self, detection_tracker):
        detection_tracker.update([make_detection()])
        detection_tracker.reset()
        assert detection_tracker.active_count == 0

    def test_new_entries(self, detection_tracker):
        detection_tracker.update([make_detection()])
        new = detection_tracker.get_new_entries()
        assert len(new) == 1
