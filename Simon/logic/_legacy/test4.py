import time
from detection_schema import Detection
from spatial_logic import SpatialLogic
from priority_logic import PriorityLogic
from safety_logic import SafetyLogic

FRAME_W, FRAME_H = 640, 480

spatial = SpatialLogic(FRAME_W, FRAME_H)
priority = PriorityLogic()
safety = SafetyLogic(cooldown=2)

# Two hazards
d1 = Detection(
    cls="car",
    bbox=[10, 120, 300, 460],  # near + front
    confidence=0.95
)

d2 = Detection(
    cls="motorcycle",
    bbox=[10, 120, 300, 450],  # near + left
    confidence=0.92
)

detections = [d1, d2]

# Apply spatial + priority
detections = [spatial.apply(d) for d in detections]
detections = [priority.apply(d) for d in detections]

print("Call 1:")
print(safety.check(detections))   # FIRST WARNING

time.sleep(2.2)                   # ⏱️ WAIT MORE THAN COOLDOWN

print("Call 2:")
print(safety.check(detections))   # SECOND WARNING

time.sleep(2.2)

print("Call 3:")
print(safety.check(detections))   # SHOULD BE None
