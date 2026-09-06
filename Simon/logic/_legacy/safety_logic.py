# logic/safety_logic.py
import time

class SafetyLogic:
    HAZARD_CLASSES = {
        "auto-rickshaw",
        "bicycle",
        "bus",
        "car",
        "mini_truck",
        "motorcycle",
        "scooter",
        "truck"
    }

    def __init__(self, cooldown=2.0):
        self.cooldown = cooldown
        # Per-hazard cooldown: { hazard_key: last_spoken_timestamp }
        self.spoken_hazards = {}

    def check(self, detections):
        """
        Returns ONE warning string at a time (sequentially),
        or None if no warning should be spoken now.
        """

        current_time = time.time()

        # Filter near hazards
        hazards = [
            d for d in detections
            if d.cls in self.HAZARD_CLASSES and d.distance == "near"
        ]

        if not hazards:
            # No active hazards — prune all memory
            self.spoken_hazards.clear()
            return None

        # Sort hazards by importance
        hazards.sort(
            key=lambda d: (
                d.position != "front",
                -(d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                -d.priority
            )
        )

        # Prune stale entries that have exceeded the cooldown
        self.spoken_hazards = {
            k: t for k, t in self.spoken_hazards.items()
            if current_time - t < self.cooldown
        }

        # Speak the highest-priority hazard not on cooldown
        for h in hazards:
            # Key by class + position only (not pixel coords) so that a truck
            # staying in the same region isn't re-announced every frame due to
            # minor bounding-box jitter between detections.
            hazard_key = f"{h.cls}_{h.position}"
            if hazard_key not in self.spoken_hazards:
                self.spoken_hazards[hazard_key] = current_time
                return f"Warning. A {h.cls} is very close {h.position}."

        return None
