from ultralytics import YOLO
from logic.detection_schema import Detection

class YoloDetector:
    def __init__(self, model_path="models/yolo_best.pt"):
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # your custom classes

    def detect(self, frame):
        results = self.model(frame, conf=0.45, iou=0.5, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                Detection(
                    cls=class_name,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf
                )
            )

        return detections