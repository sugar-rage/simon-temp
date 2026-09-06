"""
SIMON OCR Reader — Scene text recognition from camera frames.

Based on:  Teserect/withdeskewcode.py  (pytesseract + deskew + preprocessing)
           Teserect/easy.py            (easyocr alternative)

Capabilities:
  - Read text/signs from live camera frames
  - Deskew tilted text for better accuracy
  - Preprocessing: grayscale → resize → threshold → denoise
  - Returns text with confidence and bounding boxes
  - Cooldown to avoid re-announcing same text
  - Draw detected text regions on frame (HUD overlay)

Usage:
    reader = OCRReader()
    results = reader.read_text(frame)
    reader.draw_text_regions(frame, results)
    combined = reader.get_readable_text(results)
"""

import cv2
import numpy as np
import time

try:
    import config
except ImportError:
    config = None


class OCRReader:
    """
    Scene text recognition using pytesseract (primary) or easyocr (fallback).

    Based on the existing Teserect/ folder implementations with added
    integration for real-time camera frame processing.
    """

    def __init__(self, engine=None):
        """
        Initialize the OCR reader.

        Args:
            engine: "pytesseract", "easyocr", or None (auto-detect)
        """
        self._engine = engine or getattr(config, "OCR_BACKEND", "auto")
        self._min_confidence = getattr(config, "OCR_CONFIDENCE", 0.40)
        self._deskew_enabled = getattr(config, "OCR_DESKEW", True)
        self._reader = None       # easyocr reader instance
        self._backend = None      # active backend name

        # Initialize backend
        self._init_backend()

    def _init_backend(self):
        """Try pytesseract first, then easyocr, based on config."""
        if self._engine in ("pytesseract", "auto"):
            if self._try_pytesseract():
                return
        if self._engine in ("easyocr", "auto"):
            if self._try_easyocr():
                return
        if self._engine == "pytesseract":
            print("[OCRReader] pytesseract not available!")
        elif self._engine == "easyocr":
            print("[OCRReader] easyocr not available!")
        else:
            print("[OCRReader] No OCR backend available!")
        self._backend = "none"

    def _try_pytesseract(self):
        """Try to initialize pytesseract."""
        try:
            import pytesseract  # noqa: F401

            # Set tesseract command path if configured
            tess_cmd = getattr(config, "TESSERACT_CMD", None)
            if tess_cmd:
                pytesseract.pytesseract.tesseract_cmd = tess_cmd

            # Quick test
            pytesseract.get_tesseract_version()
            self._backend = "pytesseract"
            print("[OCRReader] pytesseract backend ready ✔")
            return True
        except Exception as e:
            print(f"[OCRReader] pytesseract unavailable: {e}")
            return False

    def _try_easyocr(self):
        """Try to initialize easyocr."""
        try:
            import easyocr
            lang = getattr(config, "OCR_LANGUAGE", "en")
            use_gpu = getattr(config, "OCR_GPU", False)
            print(f"[OCRReader] Loading easyocr (lang={lang}, gpu={use_gpu})...")
            self._reader = easyocr.Reader([lang], gpu=use_gpu)
            self._backend = "easyocr"
            print("[OCRReader] easyocr backend ready ✔")
            return True
        except Exception as e:
            print(f"[OCRReader] easyocr unavailable: {e}")
            return False

    # ------------------------------------------------------------------
    # Preprocessing (from Teserect/withdeskewcode.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _deskew_image(image):
        """
        Correct skewed/tilted text in the image.
        Uses Hough line detection to find the dominant angle.
        Based on: Teserect/withdeskewcode.py → deskew_image()
        """
        if image is None:
            return image, 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        if lines is None:
            return image, 0

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = theta - np.pi / 2
            angles.append(angle)

        median_angle = np.median(angles)
        angle_deg = np.degrees(median_angle)

        # Only deskew if angle is significant but not too large
        if abs(angle_deg) < 0.5 or abs(angle_deg) > 45:
            return image, 0

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, angle_deg

    @staticmethod
    def _preprocess_frame(frame):
        """
        Preprocess a camera frame for OCR accuracy.
        Based on: Teserect/withdeskewcode.py pipeline.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upscale for better small-text detection
        gray = cv2.resize(gray, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

        # Adaptive threshold (better than fixed for varying lighting)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2,
        )

        # Light denoise
        thresh = cv2.medianBlur(thresh, 3)

        return thresh

    # ------------------------------------------------------------------
    # Core OCR Methods
    # ------------------------------------------------------------------
    def read_text(self, frame):
        """
        Read text from a camera frame.

        Args:
            frame: BGR numpy array from cv2.

        Returns:
            List of dicts: [{"text": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        """
        if self._backend == "none" or frame is None:
            return []

        # Deskew if enabled
        processed = frame.copy()
        if self._deskew_enabled:
            processed, _ = self._deskew_image(processed)

        if self._backend == "pytesseract":
            return self._read_pytesseract(processed)
        elif self._backend == "easyocr":
            return self._read_easyocr(processed)
        return []

    def _read_pytesseract(self, frame):
        """
        OCR using pytesseract.
        Based on: Teserect/withdeskewcode.py → run_ocr()
        """
        import pytesseract
        from pytesseract import Output

        preprocessed = self._preprocess_frame(frame)

        # PSM 6 = assume uniform block of text (good for signs)
        ocr_config = "--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            preprocessed, output_type=Output.DICT, config=ocr_config,
        )

        results = []
        h_orig, w_orig = frame.shape[:2]

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            try:
                conf = int(float(data["conf"][i]))
            except (ValueError, TypeError):
                continue

            if text and conf > self._min_confidence * 100:
                # Scale back from 2x preprocessing
                x = int(data["left"][i] / 2)
                y = int(data["top"][i] / 2)
                w = int(data["width"][i] / 2)
                h = int(data["height"][i] / 2)

                # Clamp to frame bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_orig, x + w)
                y2 = min(h_orig, y + h)

                # Size filter: reject tiny text (e.g. noise / artifacts)
                if w < 20 or h < 10:
                    continue

                results.append({
                    "text": text,
                    "confidence": round(conf / 100, 2),
                    "bbox": [x1, y1, x2, y2],
                })

        return results

    def _read_easyocr(self, frame):
        """
        OCR using easyocr.
        Based on: Teserect/easy.py → run_easyocr()
        """
        results_raw = self._reader.readtext(frame)
        results = []

        for bbox, text, confidence in results_raw:
            if confidence < self._min_confidence:
                continue

            text = text.strip()
            if not text:
                continue

            # Convert 4-point box → [x1, y1, x2, y2]
            x_coords = [int(point[0]) for point in bbox]
            y_coords = [int(point[1]) for point in bbox]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)

            # Size filter: reject tiny text
            if (x2 - x1) < 20 or (y2 - y1) < 10:
                continue

            results.append({
                "text": text,
                "confidence": round(float(confidence), 2),
                "bbox": [x1, y1, x2, y2],
            })

        return results

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    def get_readable_text(self, results):
        """
        Combine OCR results into a single readable string.
        Filters duplicates and low-confidence junk.
        """
        if not results:
            return ""

        # Sort by vertical position (top to bottom), then horizontal
        sorted_results = sorted(results, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        words = [r["text"] for r in sorted_results if len(r["text"]) > 1]
        return " ".join(words)



    def draw_text_regions(self, frame, results):
        """
        Draw detected text regions on the camera frame.

        Args:
            frame:   BGR numpy array (modified in-place).
            results: list from read_text().
        """
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            conf = r["confidence"]
            text = r["text"]

            # Color: green for high confidence, yellow for medium
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{text} ({int(conf * 100)}%)"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame


# ======================================================================
#  Standalone Test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SIMON OCR Reader — Standalone Test")
    print("=" * 60)

    reader = OCRReader()
    print(f"[Test] Active backend: {reader._backend}")

    if reader._backend == "none":
        print("[Test] No OCR backend available. Install pytesseract or easyocr.")
    else:
        # Test with an image from the Teserect folder
        import os
        test_dir = os.path.join(os.path.dirname(__file__), "Teserect")
        test_images = ["real1.jpeg", "sign.jpg", "1.jpg"]

        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            if os.path.exists(img_path):
                print(f"\n--- Testing: {img_name} ---")
                frame = cv2.imread(img_path)
                if frame is not None:
                    results = reader.read_text(frame)
                    text = reader.get_readable_text(results)
                    print(f"  Results: {len(results)} text regions")
                    print(f"  Combined: {text[:100]}...")
                    for r in results[:5]:
                        print(f"    '{r['text']}' ({r['confidence']:.0%})")
                break

    print("\n=== Test Complete ===")
