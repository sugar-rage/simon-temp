# logic/logic_controller.py

from .spatial_logic import SpatialLogic
from .priority_logic import PriorityLogic
from .safety_logic import SafetyLogic
from .memory_logic import MemoryLogic
from .language_logic import LanguageLogic


class LogicController:
    def __init__(self, frame_width, frame_height):
        # Core logic modules
        self.spatial = SpatialLogic(frame_width, frame_height)
        self.priority = PriorityLogic()
        self.safety = SafetyLogic(cooldown=2.0)
        self.memory = MemoryLogic(cooldown=5.0)
        self.language = LanguageLogic()

    def process(self, detections):
        """
        detections: list of Detection objects
        returns: speech string or None
        """

        # 1️⃣ Spatial reasoning
        detections = [self.spatial.apply(d) for d in detections]

        # 2️⃣ Priority scoring
        detections = [self.priority.apply(d) for d in detections]

        # 3️⃣ SAFETY OVERRIDE (hazards)
        warning = self.safety.check(detections)
        if warning:
            return warning, 1

        # 4️⃣ Sort by priority (high → low)
        detections.sort(key=lambda d: d.priority, reverse=True)

        # 5️⃣ Context memory (non-hazard filtering)
        selected = []
        for d in detections:
            if self.memory.should_speak(d):
                selected.append(d)
            if len(selected) == 2:   # limit speech length
                break

        if not selected:
            return None

        # 6️⃣ Language generation
        speech_text = self.language.generate(selected)

        return speech_text, 5