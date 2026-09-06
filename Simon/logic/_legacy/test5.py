import time
from detection_schema import Detection
from spatial_logic import SpatialLogic
from memory_logic import MemoryLogic

FRAME_W, FRAME_H = 640, 480

spatial = SpatialLogic(FRAME_W, FRAME_H)
memory = MemoryLogic(cooldown=3)

# Same object, same position
d = Detection(
    cls="person",
    bbox=[200, 100, 420, 460],
    confidence=0.95
)

# Apply spatial logic
d = spatial.apply(d)

print("Call 1:", memory.should_speak(d))  # True
print("Call 2:", memory.should_speak(d))  # False

time.sleep(3.2)

print("Call 3:", memory.should_speak(d))  # True
