from detection_schema import Detection
from spatial_logic import SpatialLogic
from priority_logic import PriorityLogic

FRAME_W, FRAME_H = 640, 480

# Create modules
spatial = SpatialLogic(FRAME_W, FRAME_H)
priority = PriorityLogic()

# Fake detection
d = Detection(
    cls="person",
    bbox=[200, 100, 420, 450],
    confidence=0.95,
    face_id="Rahul"
)

d = spatial.apply(d)
d = priority.apply(d)

print("Position:", d.position)
print("Distance:", d.distance)
print("Priority:", d.priority)
