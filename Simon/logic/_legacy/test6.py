from detection_schema import Detection
from spatial_logic import SpatialLogic
from priority_logic import PriorityLogic
from language_logic import LanguageLogic

FRAME_W, FRAME_H = 640, 480

spatial = SpatialLogic(FRAME_W, FRAME_H)
priority = PriorityLogic()
language = LanguageLogic()

# Example detections
d1 = Detection(
    cls="person",
    bbox=[200, 120, 420, 460],
    confidence=0.95,
    face_id="Rahul"
)

d2 = Detection(
    cls="traffic_signal",
    bbox=[500, 100, 620, 300],
    confidence=0.88
)

detections = [d1, d2]

# Apply logic
detections = [spatial.apply(d) for d in detections]
detections = [priority.apply(d) for d in detections]

# Only top 2 (as logic controller will do later)
detections = sorted(detections, key=lambda d: d.priority, reverse=True)[:2]

speech = language.generate(detections)

print(speech)
