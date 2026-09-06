# logic/priority_logic.py

class PriorityLogic:

    BASE_PRIORITY = {
        # Moving hazards
        "auto-rickshaw": 5,
        "bicycle": 5,
        "bus": 5,
        "car": 5,
        "mini_truck": 5,
        "motorcycle": 5,
        "scooter": 5,
        "truck": 5,

        # Human
        "person": 4,

        # Traffic infrastructure
        "traffic_signal": 3,
        "signal_pole": 3,
        "street_light": 3,

        # Perception extensions
        "face": 4,
        "text": 3
    }

    def apply(self, det):
        score = self.BASE_PRIORITY.get(det.cls, 1)

        # Distance importance
        if det.distance == "near":
            score += 2

        # Direction importance
        if det.position == "front":
            score += 2

        # Identity awareness
        if det.cls == "person" and det.face_id:
            score += 2
        elif det.cls == "face" and det.face_id != "Unknown":
            score += 2

        det.priority = score
        return det
