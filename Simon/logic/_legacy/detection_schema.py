# logic/detection_schema.py

class Detection:
    def __init__(self, cls, bbox, confidence,
                 face_id=None, text=None):
        self.cls = cls              # object class name
        self.bbox = bbox            # [x1, y1, x2, y2]
        self.confidence = confidence

        # Optional perception outputs
        self.face_id = face_id      # recognized name or None
        self.text = text            # OCR text or None

        # Logic layer fields (filled later)
        self.position = None        # left / right / front
        self.distance = None        # near / medium / far
        self.priority = None
