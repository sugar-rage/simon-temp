from detection_schema import Detection
from logic_controller import LogicController

FRAME_W, FRAME_H = 640, 480

logic = LogicController(FRAME_W, FRAME_H)

# Example detections
detections = [
    Detection(
        cls="person",
        bbox=[200, 120, 420, 460],
        confidence=0.95,
        face_id="Rahul"
    ),
    Detection(
        cls="car",
        bbox=[10, 120, 300, 450],
        confidence=0.88
    )
]

speech = logic.process(detections)
print("Output:", speech)
