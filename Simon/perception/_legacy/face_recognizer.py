"""
SIMON Face Recognizer — InsightFace-based face detection and recognition.

Uses antelopev2 = RetinaFace detector + ArcFace R100 (glintr100) recognizer.

Capabilities:
  - Detect and recognize faces in real-time camera frames
  - Folder-based identity storage (SAVED_FACES/<name>/0.jpg..9.jpg)
  - Multi-embedding per person (up to 10) for robust recognition
  - Auto-learning: adds new embeddings when score > threshold
  - Duplicate detection: skips if embedding is >95% similar
  - Migration: auto-converts old flat-file layout to folder-based

Usage:
    sfr = SimpleFacerec()
    face_locations, face_names, face_objects, face_scores = sfr.detect_known_faces(frame)
    sfr.save_face(frame, face_obj, "person_name")
"""

import cv2
import os
import shutil
import time
import numpy as np
from insightface.app import FaceAnalysis


class SimpleFacerec:
    # --- Configuration Constants ---
    MIN_FACE_SIZE = 40              # Ignore faces smaller than this (pixels)
    DUPLICATE_THRESHOLD = 0.95     # Skip auto-learn if embedding is this similar
    MAX_EMBEDDINGS_PER_PERSON = 10  # Max face samples per identity
    STABILITY_BUFFER_FRAMES = 5    # Frames to hold identity before declaring Unknown
    AUTO_LEARN_THRESHOLD = 0.50    # Auto-learn if score is above this but below recognition
    MAX_STABILITY_ENTRIES = 50     # Hard cap on stability buffer size
    STABILITY_EXPIRE_SECONDS = 10.0  # Auto-expire entries not seen for this long

    def __init__(self, model_name="antelopev2", det_size=(640, 640)):
        """
        Initialize with separate detection and recognition models.
        Uses antelopev2 = RetinaFace detector + ArcFace R100 (glintr100) recognizer.
        """
        print(f"Loading InsightFace model ({model_name})...")

        import insightface
        import onnxruntime as ort
        import numpy as np

        print(f"InsightFace version: {getattr(insightface, '__version__', 'unknown')}")
        print(f"ONNX Runtime version: {getattr(ort, '__version__', 'unknown')}")
        print(f"NumPy version: {getattr(np, '__version__', 'unknown')}")
        
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        # Safely fallback to CPU if CUDA is not fully configured
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']

        # FaceAnalysis will download the model if it doesn't exist.
        self.app = FaceAnalysis(name=model_name, providers=providers)
        
        # Auto-fix: some InsightFace zips extract into a nested subdirectory
        # e.g. antelopev2/antelopev2/*.onnx instead of antelopev2/*.onnx
        # We must call this AFTER FaceAnalysis so it fixes newly downloaded models!
        self._fix_model_directory(model_name)

        # Now prepare the models
        self.app.prepare(ctx_id=0, det_size=det_size)

        # Identify detector and recognizer from loaded models
        self.detector = None
        self.recognizer = None
        for model in self.app.models:
            if hasattr(model, 'taskname'):
                if model.taskname == 'detection':
                    self.detector = model
                    print(f"  Detector: {type(model).__name__}")
                elif model.taskname == 'recognition':
                    self.recognizer = model
                    print(f"  Recognizer: {type(model).__name__}")

        if self.detector is None:
            print("  Warning: Could not identify detector model separately.")
        if self.recognizer is None:
            print("  Warning: Could not identify recognizer model separately.")

        self.known_embeddings = {}
        self._stability_buffer = {}   # bbox → {name, miss_count}
        try:
            import config
            self.RECOGNITION_THRESHOLD = getattr(config, "FACE_MATCH_THRESHOLD", 0.58)
            self.save_folder = getattr(config, "FACE_DATABASE_PATH", "SAVED_FACES")
        except ImportError:
            self.RECOGNITION_THRESHOLD = 0.58
            self.save_folder = "SAVED_FACES"

        # Determine absolute path for saving
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(base_dir, self.save_folder)
        os.makedirs(self.save_path, exist_ok=True)

        # Migrate old flat-file structure if needed
        self._migrate_flat_files()

        # Load all embeddings at startup
        self.load_encoding_images()

    # ------------------------------------------------------------------
    # Model directory fix (nested zip extraction)
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_model_directory(model_name):
        """
        Fix nested extraction: e.g. models/antelopev2/antelopev2/*.onnx
        should be models/antelopev2/*.onnx
        """
        home = os.path.expanduser("~")
        model_dir = os.path.join(home, ".insightface", "models", model_name)
        nested_dir = os.path.join(model_dir, model_name)

        if os.path.isdir(nested_dir):
            # Check if nested dir has ONNX files but parent doesn't
            parent_onnx = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            nested_onnx = [f for f in os.listdir(nested_dir) if f.endswith('.onnx')]

            if not parent_onnx and nested_onnx:
                print(f"  [Fix] Moving model files from nested {model_name}/{model_name}/ ...")
                for f in os.listdir(nested_dir):
                    src = os.path.join(nested_dir, f)
                    dst = os.path.join(model_dir, f)
                    shutil.move(src, dst)
                shutil.rmtree(nested_dir, ignore_errors=True)
                print(f"  [Fix] Model directory structure corrected.")

    # ------------------------------------------------------------------
    # Migration: flat files -> folder-based structure
    # ------------------------------------------------------------------
    def _migrate_flat_files(self):
        """
        Auto-migrate old flat-file layout (name_0.jpg) into folder-based layout (name/0.jpg).
        Runs once at startup; safe to call multiple times.
        """
        migrated = 0
        for file in os.listdir(self.save_path):
            filepath = os.path.join(self.save_path, file)
            if not os.path.isfile(filepath):
                continue
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Parse name from pattern "name_0.jpg"
            base = os.path.splitext(file)[0]      # "name_0"
            parts = base.rsplit("_", 1)            # ["name", "0"]
            if len(parts) == 2 and parts[1].isdigit():
                person_name = parts[0]
                index = parts[1]
            else:
                person_name = base
                index = "0"

            person_dir = os.path.join(self.save_path, person_name)
            os.makedirs(person_dir, exist_ok=True)

            ext = os.path.splitext(file)[1]
            dest = os.path.join(person_dir, f"{index}{ext}")
            shutil.move(filepath, dest)
            migrated += 1
            print(f"  [Migration] {file} -> {person_name}/{index}{ext}")

        if migrated > 0:
            print(f"  [Migration] Migrated {migrated} file(s) to folder-based layout.")

    # ------------------------------------------------------------------
    # Loading: folder-based identity structure
    # ------------------------------------------------------------------
    def load_encoding_images(self):
        """
        Loads images from SAVED_FACES/<person_name>/ folders and extracts embeddings.
        Each subfolder = one identity. All images inside = multiple embeddings.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print("[FaceRecognizer] No registered faces found (directory created).")
            return

        people = os.listdir(self.save_path)
        if not people:
            print("[FaceRecognizer] No registered faces found.")
            return

        print(f"Loading known faces from: {self.save_path}")
        self.known_embeddings = {}

        for person_name in sorted(os.listdir(self.save_path)):
            person_dir = os.path.join(self.save_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []
            for img_file in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_file)
                if not os.path.isfile(img_path):
                    continue
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Add padding to help detection on tightly-cropped saved faces
                img_padded = cv2.copyMakeBorder(
                    img, 50, 50, 50, 50,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

                faces = self.app.get(img_padded)
                if len(faces) == 0:
                    faces = self.app.get(img)  # Fallback to original

                if len(faces) > 0:
                    embedding = faces[0].embedding
                    norm = np.linalg.norm(embedding)
                    if norm != 0:
                        embedding = embedding / norm
                    embeddings.append(embedding)
                else:
                    print(f"  Warning: No face found in {person_name}/{img_file}")

            if embeddings:
                self.known_embeddings[person_name] = embeddings
                print(f"  Loaded: {person_name} ({len(embeddings)} embedding(s))")

        print(f"Total known people: {len(self.known_embeddings)}")
        self._update_matrix()

    def _update_matrix(self):
        """Builds a vectorized 2D matrix of all known embeddings for fast matching."""
        names = []
        embs = []
        for name, known_embs in self.known_embeddings.items():
            names.extend([name] * len(known_embs))
            embs.extend(known_embs)
        
        self._emb_names = names
        self._emb_matrix = np.array(embs) if embs else np.empty((0, 0))

    # ------------------------------------------------------------------
    # Detection + Recognition
    # ------------------------------------------------------------------
    def detect_known_faces(self, frame, recognition_threshold=None):
        """
        Detects faces in the frame and identifies them.

        Pipeline:
            Frame -> Face Detector (RetinaFace) -> Face Crop
                  -> Recognition Model (ArcFace R100) -> Embedding Comparison

        Includes a stability buffer: a face recognized as "Person X" will
        keep that label for STABILITY_BUFFER_FRAMES even if recognition
        momentarily drops (e.g. due to motion blur).

        Returns:
            face_locations: List of (x1, y1, x2, y2)
            face_names:     List of names (or "Unknown")
            face_objects:   List of InsightFace face objects
            face_scores:    List of cosine similarity scores
        """
        if recognition_threshold is None:
            recognition_threshold = self.RECOGNITION_THRESHOLD

        # Prune stale stability buffer entries before processing
        self._cleanup_stability_buffer()

        faces = self.app.get(frame)

        face_locations = []
        face_names = []
        face_scores = []
        filtered_faces = []

        h, w, _ = frame.shape

        for face in faces:
            # 1. Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # Clamp to frame
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            # 2. Skip very small faces
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.MIN_FACE_SIZE or face_h < self.MIN_FACE_SIZE:
                continue

            face_locations.append((x1, y1, x2, y2))
            filtered_faces.append(face)

            # 3. Recognition: normalize embedding
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm != 0:
                embedding = embedding / norm

            # 4. Compare against ALL stored embeddings (Vectorized)
            best_name = "Unknown"
            max_score = -1.0

            if hasattr(self, '_emb_matrix') and self._emb_matrix.size > 0:
                scores = np.dot(self._emb_matrix, embedding)
                best_idx = np.argmax(scores)
                max_score = scores[best_idx]
                if max_score > recognition_threshold:
                    best_name = self._emb_names[best_idx]

            # 5. Stability buffer: prevent "Unknown" flicker
            best_name = self._apply_stability_buffer(
                (x1, y1, x2, y2), best_name, max_score
            )

            face_names.append(best_name)
            face_scores.append(max_score)

        return face_locations, face_names, filtered_faces, face_scores

    # ------------------------------------------------------------------
    # Stability Buffer
    # ------------------------------------------------------------------
    def _apply_stability_buffer(self, bbox, raw_name, score):
        """
        Prevent flickering between a known name and 'Unknown'.

        If a face WAS recognized as someone but this frame it's Unknown,
        keep the old name for STABILITY_BUFFER_FRAMES consecutive misses.
        This smooths out motion blur / angle changes.
        """
        # Find matching region in buffer (IoU-based)
        buf_key = self._find_matching_region(bbox)
        now = time.time()

        if raw_name != "Unknown":
            # Recognized — update buffer with this identity
            self._stability_buffer[buf_key] = {
                "name": raw_name,
                "miss_count": 0,
                "last_seen": now,
            }
            return raw_name

        # Not recognized — check if we have a recent identity for this region
        if buf_key in self._stability_buffer:
            entry = self._stability_buffer[buf_key]
            entry["miss_count"] += 1
            entry["last_seen"] = now

            if entry["miss_count"] <= self.STABILITY_BUFFER_FRAMES:
                # Still within buffer — keep the old name
                return entry["name"]
            else:
                # Too many misses — truly Unknown
                del self._stability_buffer[buf_key]

        return "Unknown"

    def _cleanup_stability_buffer(self):
        """
        Remove stale entries from the stability buffer to prevent
        unbounded memory growth during long-running operation.

        Two-phase cleanup:
          1. Expire entries not seen for STABILITY_EXPIRE_SECONDS
          2. If still over MAX_STABILITY_ENTRIES, evict oldest entries
        """
        now = time.time()

        # Phase 1: remove expired entries
        expired_keys = [
            k for k, v in self._stability_buffer.items()
            if now - v.get("last_seen", 0) > self.STABILITY_EXPIRE_SECONDS
        ]
        for k in expired_keys:
            del self._stability_buffer[k]

        # Phase 2: enforce hard cap (evict oldest if still over limit)
        if len(self._stability_buffer) > self.MAX_STABILITY_ENTRIES:
            sorted_entries = sorted(
                self._stability_buffer.items(),
                key=lambda item: item[1].get("last_seen", 0)
            )
            to_remove = len(self._stability_buffer) - self.MAX_STABILITY_ENTRIES
            for k, _ in sorted_entries[:to_remove]:
                del self._stability_buffer[k]

    def _find_matching_region(self, bbox):
        """Find the buffer key that overlaps most with this bbox."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        best_key = None
        best_dist = float('inf')

        for key in list(self._stability_buffer.keys()):
            kx1, ky1, kx2, ky2 = key
            kcx, kcy = (kx1 + kx2) // 2, (ky1 + ky2) // 2
            dist = abs(cx - kcx) + abs(cy - kcy)
            if dist < best_dist and dist < 100:  # within 100px
                best_dist = dist
                best_key = key

        return best_key if best_key else (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Auto-Learning
    # ------------------------------------------------------------------
    def add_live_embedding(self, name, embedding, frame=None, face_obj=None):
        """
        Adds a new embedding for an existing person if it's sufficiently different.
        Also saves the face crop to disk for persistence across restarts.

        Args:
            name:      Person name
            embedding: Raw embedding from face object
            frame:     Current video frame (optional, for saving crop)
            face_obj:  InsightFace face object (optional, for saving crop)

        Returns:
            True if a new embedding was added, False if duplicate.
        """
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm

        if name not in self.known_embeddings:
            self.known_embeddings[name] = []

        # Check for duplicates
        for existing_emb in self.known_embeddings[name]:
            score = np.dot(embedding, existing_emb)
            if score > self.DUPLICATE_THRESHOLD:
                return False  # Too similar, skip

        self.known_embeddings[name].append(embedding)
        count = len(self.known_embeddings[name])
        print(f"[Auto-Learn] Added new embedding for {name}. Total: {count}")

        # Enforce limit (FIFO)
        if count > self.MAX_EMBEDDINGS_PER_PERSON:
            self.known_embeddings[name].pop(0)
            print(f"[Auto-Learn] Removed oldest embedding for {name} (limit reached)")

        # Save face crop to disk for persistence
        if frame is not None and face_obj is not None:
            self._save_auto_learn_crop(name, frame, face_obj)

        self._update_matrix()
        return True

    def _save_auto_learn_crop(self, name, frame, face_obj):
        """Save an auto-learned face crop to the person's folder."""
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (' ', '_', '-')
        ).strip()
        if not safe_name:
            return

        person_dir = os.path.join(self.save_path, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        # Get crop
        bbox = face_obj.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        h, w, _ = frame.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        crop = frame[y1:y2, x1:x2].copy()

        # Find next available slot or overwrite oldest (FIFO)
        saved = False
        for i in range(self.MAX_EMBEDDINGS_PER_PERSON):
            path = os.path.join(person_dir, f"{i}.jpg")
            if not os.path.exists(path):
                cv2.imwrite(path, crop)
                saved = True
                break

        if not saved:
            # All slots full — find oldest file and overwrite it
            oldest_path = None
            oldest_time = float('inf')
            for i in range(self.MAX_EMBEDDINGS_PER_PERSON):
                path = os.path.join(person_dir, f"{i}.jpg")
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if mtime < oldest_time:
                        oldest_time = mtime
                        oldest_path = path
            if oldest_path:
                cv2.imwrite(oldest_path, crop)

    # ------------------------------------------------------------------
    # Save Face (manual, user-triggered)
    # ------------------------------------------------------------------
    def save_face(self, frame, face_obj, name):
        """
        Saves a face to disk (folder-based) and adds to memory immediately.

        Layout: SAVED_FACES/<name>/0.jpg .. 9.jpg
        """
        # 1. Clean name
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (' ', '_', '-')
        ).strip()
        if not safe_name:
            print("Invalid name.")
            return False

        # 2. Ensure person folder exists
        person_dir = os.path.join(self.save_path, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        # 3. Get crop
        bbox = face_obj.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        h, w, _ = frame.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        crop = frame[y1:y2, x1:x2].copy()

        # 4. Save to disk — find available slot or overwrite oldest (FIFO)
        saved_path = ""
        msg_action = "Saved"

        for i in range(self.MAX_EMBEDDINGS_PER_PERSON):
            path = os.path.join(person_dir, f"{i}.jpg")
            if not os.path.exists(path):
                saved_path = path
                break

        if not saved_path:
            # All slots full — overwrite oldest
            oldest_path = None
            oldest_time = float('inf')
            for i in range(self.MAX_EMBEDDINGS_PER_PERSON):
                path = os.path.join(person_dir, f"{i}.jpg")
                if os.path.exists(path):
                    mtime = os.path.getmtime(path)
                    if mtime < oldest_time:
                        oldest_time = mtime
                        oldest_path = path
            saved_path = oldest_path
            msg_action = "Overwrote (FIFO)"

        cv2.imwrite(saved_path, crop)
        print(f"{msg_action}: {saved_path}")

        # 5. Add to memory immediately
        embedding = face_obj.embedding.copy()
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm

        if safe_name not in self.known_embeddings:
            self.known_embeddings[safe_name] = []

        self.known_embeddings[safe_name].append(embedding)

        # Enforce limit in memory too
        if len(self.known_embeddings[safe_name]) > self.MAX_EMBEDDINGS_PER_PERSON:
            self.known_embeddings[safe_name].pop(0)

        print(f"Added {safe_name} to live memory. "
              f"Total embeddings: {len(self.known_embeddings[safe_name])}")

        self._update_matrix()
        return True
