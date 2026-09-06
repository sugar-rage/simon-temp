# logic/spatial_logic.py

class SpatialLogic:
    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height

    def apply(self, det):
        x1, y1, x2, y2 = det.bbox

        # Center X of bounding box
        cx = (x1 + x2) / 2

        # Area of bounding box
        area = (x2 - x1) * (y2 - y1)
        frame_area = self.w * self.h

        # Position logic
        if cx < self.w * 0.33:
            det.position = "left"
        elif cx > self.w * 0.66:
            det.position = "right"
        else:
            det.position = "front"

        # Distance logic (approximation)
        if area > frame_area * 0.15:
            det.distance = "near"
        elif area > frame_area * 0.05:
            det.distance = "medium"
        else:
            det.distance = "far"

        return det