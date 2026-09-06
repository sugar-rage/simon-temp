"""Unit tests for SceneAnalyzer — detection-based scene classification."""

from __future__ import annotations

import pytest

from vision.scene.scene_analyzer import SceneAnalyzer
from vision.tests.conftest import make_detection


class TestSceneAnalyzer:
    def test_empty_detections(self, scene_analyzer):
        result = scene_analyzer.analyze([])
        assert result.scene_type == "unknown"
        assert result.confidence == 0.0

    def test_street_scene(self, scene_analyzer):
        dets = [
            make_detection(cls="car"),
            make_detection(cls="car", bbox=(200, 200, 300, 300)),
            make_detection(cls="truck", bbox=(300, 300, 400, 400)),
            make_detection(cls="traffic light", bbox=(400, 10, 420, 50)),
        ]
        result = scene_analyzer.analyze(dets)
        assert result.scene_type == "street"
        assert result.confidence > 0.0

    def test_indoor_scene(self, scene_analyzer):
        dets = [
            make_detection(cls="chair"),
            make_detection(cls="couch", bbox=(200, 200, 300, 300)),
            make_detection(cls="tv", bbox=(300, 300, 400, 400)),
        ]
        result = scene_analyzer.analyze(dets)
        assert result.scene_type == "indoor"

    def test_park_scene(self, scene_analyzer):
        dets = [
            make_detection(cls="dog"),
            make_detection(cls="bench", bbox=(200, 200, 300, 300)),
            make_detection(cls="bird", bbox=(300, 300, 350, 350)),
        ]
        result = scene_analyzer.analyze(dets)
        assert result.scene_type == "park"

    def test_person_only_is_outdoor(self, scene_analyzer):
        dets = [make_detection(cls="person")]
        result = scene_analyzer.analyze(dets)
        assert result.scene_type == "outdoor"

    def test_description_contains_objects(self, scene_analyzer):
        dets = [make_detection(cls="car"), make_detection(cls="person", bbox=(200, 200, 300, 300))]
        result = scene_analyzer.analyze(dets)
        assert "car" in result.description.lower()

    def test_object_counts(self, scene_analyzer):
        dets = [
            make_detection(cls="car"),
            make_detection(cls="car", bbox=(200, 200, 300, 300)),
            make_detection(cls="person", bbox=(400, 400, 500, 500)),
        ]
        result = scene_analyzer.analyze(dets)
        assert result.object_counts["car"] == 2
        assert result.object_counts["person"] == 1
