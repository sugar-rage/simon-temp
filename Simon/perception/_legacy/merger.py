"""
SIMON Perception Merger — Unified pipeline for all perception modules.

Integrates:
  - object_detector.py  → Object/obstacle detection (YOLOv8)
  - face_recognizer.py  → Face detection & recognition (InsightFace)
  - ocr_reader.py       → Scene text recognition (Whisper/EasyOCR)

Runs all three modules on each camera frame, merges results into a single
PerceptionResult object that the logic layer consumes.

Usage:
    merger = PerceptionMerger()
    result = merger.process(frame)

    result.detections   → List[Detection]   (objects/obstacles)
    result.faces        → List[Detection]   (recognized faces mapped to Detection)
    result.ocr          → List[Detection]   (detected text mapped to Detection)
    result.warnings     → List[str]         (voice-ready warnings)
    result.frame        → np.ndarray        (annotated frame with HUD)
"""

import time
import cv2
import threading
import queue

try:
    import config
except ImportError:
    config = None


# ======================================================================
#  OCR Background Worker
# ======================================================================
class OCRWorker:
    """Background thread to run OCR without blocking the perception loop."""
    def __init__(self, ocr_reader, shutdown_event=None):
        self._ocr = ocr_reader
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._last_ocr_results = []
        self._shutdown_event = shutdown_event
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        """Submit a frame for OCR, dropping old ones if queue is full."""
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_latest_results(self):
        """Thread-safe read of the latest OCR raw results list."""
        with self._lock:
            return self._last_ocr_results

    def stop(self):
        self._running = False
        # Unblock queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self):
        while self._running:
            if self._shutdown_event and self._shutdown_event.is_set():
                break
            try:
                frame = self._queue.get(timeout=0.5)
                if frame is None:
                    break
                try:
                    ocr_results = self._ocr.read_text(frame)
                    if ocr_results:
                        with self._lock:
                            self._last_ocr_results = ocr_results
                except Exception as e:
                    print(f"[OCRWorker] OCR error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[OCRWorker] Worker error: {e}")

# ======================================================================
#  Perception Result — Single frame output
# ======================================================================
class FaceResult:
    """Holds a single recognized face's data."""

    __slots__ = ["name", "bbox", "score", "face_obj"]

    def __init__(self, name, bbox, score, face_obj=None):
        self.name = name
        self.bbox = bbox          # (x1, y1, x2, y2)
        self.score = score        # Cosine similarity
        self.face_obj = face_obj  # InsightFace face object (for saving)

    def __repr__(self):
        return f"FaceResult({self.name}, score={self.score:.2f})"


class PerceptionResult:
    """Aggregated output from all perception modules for a single frame."""

    def __init__(self):
        self.detections = []      # List[Detection] from object_detector
        self.faces = []           # List[Detection] with cls='face'
        self.ocr = []             # List[Detection] with cls='text'
        self.warnings = []        # Voice-ready warning strings
        self.frame = None         # Annotated frame (with HUD drawn)
        self.timestamp = 0.0      # Frame processing timestamp
        self.processing_ms = 0.0  # Total processing time in ms

    @property
    def has_obstacles(self):
        """True if any obstacles detected."""
        return len(self.detections) > 0

    @property
    def known_faces(self):
        """List of recognized (non-Unknown) faces."""
        return [f for f in self.faces if f.name != "Unknown"]

    @property
    def unknown_faces(self):
        """List of unrecognized faces."""
        return [f for f in self.faces if f.name == "Unknown"]

    @property
    def has_text(self):
        """True if OCR detected readable text."""
        return len(self.ocr) > 0

    def summary(self):
        """One-line summary for logging."""
        parts = []
        if self.detections:
            parts.append(f"{len(self.detections)} objects")
        if self.faces:
            parts.append(f"{len(self.faces)} faces")
        if self.has_text:
            parts.append(f"{len(self.ocr)} text regions")
        return " | ".join(parts) if parts else "nothing detected"


# ======================================================================
#  Perception Merger
# ======================================================================
class PerceptionMerger:
    """
    Unified perception pipeline — runs object detection, face recognition,
    and OCR on each frame and merges the results.

    Each module can be individually enabled/disabled and runs with its
    own frame skip interval for performance tuning.
    """

    def __init__(
        self,
        object_detector=None,
        face_recognizer=None,
        ocr_reader=None,
        shutdown_event=None,
    ):
        """
        Initialize the merger with optional pre-built module instances.
        If None, modules will be auto-initialized asynchronously.

        Args:
            object_detector: ObjectDetector instance (or None to auto-create)
            face_recognizer: SimpleFacerec instance (or None to auto-create)
            ocr_reader:      OCRReader instance (or None to auto-create)
            shutdown_event:  threading.Event to signal clean termination
        """
        # Module instances
        self._detector = object_detector
        self._face_rec = face_recognizer
        self._ocr = ocr_reader
        self._shutdown_event = shutdown_event

        # Enable/disable flags
        self.detect_objects = getattr(config, "OBSTACLE_ENABLED", True)
        self.detect_faces = getattr(config, "FACE_RECOGNITION_ENABLED", True)
        self.detect_text = getattr(config, "OCR_ENABLED", True)

        # Frame skip intervals (run every N frames to save CPU)
        self._frame_count = 0
        self.object_interval = 1     # Every frame
        self.face_interval = getattr(config, "FACE_RECOGNITION_INTERVAL", 5)
        self.ocr_interval = getattr(config, "OCR_INTERVAL", 10)

        # Cache last results for skipped frames
        self._last_detections = []
        self._last_faces = []
        self._last_ocr = []

        # Draw settings
        self.draw_hud = True
        self._ocr_worker = None
        
        # Async initialization tracking
        self._init_started = False
        self._init_lock = threading.Lock()

        # Start async initialization
        self._init_modules_async()

    def _init_modules_async(self):
        """Launch background thread for heavy module loading."""
        with self._init_lock:
            if self._init_started:
                return
            self._init_started = True

        threading.Thread(target=self._init_modules_worker, daemon=True).start()

    def _init_modules_worker(self):
        """Worker thread for lazy-initializing modules."""
        if self._detector is None and self.detect_objects:
            if self._shutdown_event and self._shutdown_event.is_set(): return
            try:
                from perception.yolo_detector import YoloDetector
                temp_det = YoloDetector()
                self._detector = temp_det
            except Exception as e:
                import traceback
                print(f"[Merger] Object detector init failed: {e}")
                traceback.print_exc()
                self.detect_objects = False

        if self._face_rec is None and self.detect_faces:
            if self._shutdown_event and self._shutdown_event.is_set(): return
            try:
                from perception.face_recognizer import SimpleFacerec
                temp_face = SimpleFacerec()
                self._face_rec = temp_face
            except Exception as e:
                import traceback
                print(f"[Merger] Face recognizer init failed: {e}")
                traceback.print_exc()
                self.detect_faces = False

        if self._ocr is None and self.detect_text:
            if self._shutdown_event and self._shutdown_event.is_set(): return
            try:
                from perception.ocr_reader import OCRReader
                temp_ocr = OCRReader()
                self._ocr = temp_ocr
            except Exception as e:
                import traceback
                print(f"[Merger] OCR reader init failed: {e}")
                traceback.print_exc()
                self.detect_text = False

        # Start OCR worker if OCR is enabled and module loaded
        if self.detect_text and self._ocr is not None and getattr(self, "_ocr_worker", None) is None:
            self._ocr_worker = OCRWorker(self._ocr, self._shutdown_event)

    # Old _init_modules removed.

    # ------------------------------------------------------------------
    # Main Processing Pipeline
    # ------------------------------------------------------------------
    def process(self, frame):
        """
        Run all perception modules on a frame and merge results.

        Args:
            frame: BGR numpy array from camera.

        Returns:
            PerceptionResult with all detections, faces, text, and warnings.
        """
        t0 = time.time()
        result = PerceptionResult()
        result.timestamp = t0
        result.frame = frame.copy()
        self._frame_count += 1

        # --- 1. Object Detection ---
        if self.detect_objects and self._detector:
            if self._frame_count % self.object_interval == 0:
                try:
                    self._last_detections = self._detector.detect(frame)
                except Exception as e:
                    print(f"[Merger] Object detection error: {e}")
            result.detections = self._last_detections

        # --- 2. Face Recognition ---
        if self.detect_faces and self._face_rec:
            if self._frame_count % self.face_interval == 0:
                try:
                    locs, names, objs, scores = self._face_rec.detect_known_faces(frame)
                    from logic.detection_schema import Detection
                    self._last_faces = [
                        Detection(cls="face", bbox=bbox, confidence=score, face_id=name)
                        for bbox, name, score in zip(locs, names, scores)
                    ]
                except Exception as e:
                    print(f"[Merger] Face recognition error: {e}")
            result.faces = self._last_faces

        # --- 3. OCR ---
        if self.detect_text and getattr(self, "_ocr_worker", None):
            if self._frame_count % self.ocr_interval == 0:
                # Submit frame asynchronously to prevent blocking
                self._ocr_worker.submit_frame(frame.copy())
            
            # Read whatever the latest cached results are without waiting
            latest_ocr_raw = self._ocr_worker.get_latest_results()
            if latest_ocr_raw:
                try:
                    from logic.detection_schema import Detection
                    self._last_ocr = [
                        Detection(cls="text", bbox=r["bbox"], confidence=r["confidence"], text=r["text"])
                        for r in latest_ocr_raw
                    ]
                except Exception as e:
                    print(f"[Merger] OCR parse error: {e}")
            else:
                self._last_ocr = []
            result.ocr = self._last_ocr

        # --- 4. Collect Warnings ---
        result.warnings = self._generate_warnings(result)

        # --- 5. Draw HUD ---
        if self.draw_hud:
            self._draw_hud(result)

        result.processing_ms = (time.time() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Warning Generation
    # ------------------------------------------------------------------
    def _generate_warnings(self, result):
        """Combine warnings from all modules."""
        warnings = []

        # Object warnings are handled by the logic controller
        # (spatial_logic, safety_logic, etc.) — not generated here.

        # Face announcements are handled by the logic controller
        # with proper cooldown — not generated here per-frame.

        return warnings

    # ------------------------------------------------------------------
    # HUD Drawing
    # ------------------------------------------------------------------
    def _draw_hud(self, result):
        """Draw all perception results onto the frame."""
        frame = result.frame

        # Draw object detections
        if self._detector and result.detections:
            for det in result.detections:
                x1, y1, x2, y2 = det.bbox
                color = (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.cls} {det.confidence:.0%}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw face boxes
        for face in result.faces:
            x1, y1, x2, y2 = face.bbox
            if face.face_id == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{face.face_id} ({int(face.confidence * 100)}%)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw OCR text
        if result.has_text:
            for text_det in result.ocr:
                tx1, ty1, tx2, ty2 = text_det.bbox
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
                cv2.putText(frame, text_det.text[:20], (tx1, ty1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw Readiness Indicators
        y_offset = 50
        for name, is_ready, expected in [
            ("YOLO", self.is_detector_ready, self.detect_objects),
            ("FaceRec", self.is_face_ready, self.detect_faces),
            ("OCR", self.is_ocr_ready, self.detect_text)
        ]:
            if not expected: continue
            status_text = "Ready" if is_ready else "Loading..."
            color = (0, 255, 0) if is_ready else (0, 165, 255)
            cv2.putText(frame, f"{name}: {status_text}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        # Draw FPS / processing time
        cv2.putText(frame, f"{result.processing_ms:.0f}ms",
                    (frame.shape[1] - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ------------------------------------------------------------------
    # Module Access (for main.py to use)
    # ------------------------------------------------------------------
    @property
    def face_recognizer(self):
        """Access the underlying face recognizer (for save_face, etc.)."""
        return self._face_rec

    @property
    def object_detector(self):
        """Access the underlying object detector."""
        return self._detector

    @property
    def ocr_reader(self):
        """Access the underlying OCR reader."""
        return self._ocr
        
    @property
    def is_detector_ready(self):
        return self._detector is not None
        
    @property
    def is_face_ready(self):
        return self._face_rec is not None
        
    @property
    def is_ocr_ready(self):
        return self._ocr is not None

    def stop(self):
        """Clean shutdown of background workers."""
        if self._ocr_worker:
            self._ocr_worker.stop()


# ======================================================================
#  Standalone Test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SIMON Perception Merger — Webcam Test")
    print("  Press ESC to quit")
    print("=" * 60)

    merger = PerceptionMerger()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = merger.process(frame)

        # Print warnings
        for w in result.warnings:
            print(f"[WARNING] {w}")

        # Show annotated frame
        cv2.imshow("Perception Merger", result.frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
