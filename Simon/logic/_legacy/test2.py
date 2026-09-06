from detection_schema import Detection
from spatial_logic import SpatialLogic

# Fake camera resolution
FRAME_W = 640
FRAME_H = 480

spatial = SpatialLogic(FRAME_W, FRAME_H)

# Fake detection
d = Detection(
    cls="person",
    bbox=[200, 100, 400, 450],  # centered & large box
    confidence=0.9
)

d = spatial.apply(d)

print("Position:", d.position)
print("Distance:", d.distance)
