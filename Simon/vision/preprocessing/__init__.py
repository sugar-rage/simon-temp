"""Image preprocessing transforms — stateless functions for vision pipeline.

Consolidates ``image_enhancer.py`` and ``roi_extractor.py`` per TDR-005
file structure consolidation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("simon.vision.preprocessing")


def resize_frame(
    frame: Any,
    width: int = 640,
    height: int = 480,
) -> Any:
    """Resize a frame to target dimensions."""
    try:
        import cv2

        return cv2.resize(frame, (width, height))
    except Exception:
        return frame


def to_grayscale(frame: Any) -> Any:
    """Convert a BGR frame to grayscale."""
    try:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return frame


def enhance_contrast(frame: Any) -> Any:
    """Apply CLAHE contrast enhancement to a grayscale frame."""
    try:
        import cv2

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return frame


def denoise(frame: Any, strength: int = 10) -> Any:
    """Apply non-local means denoising."""
    try:
        import cv2

        if len(frame.shape) == 2:
            return cv2.fastNlMeansDenoising(frame, None, strength, 7, 21)
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
    except Exception:
        return frame


def extract_roi(
    frame: Any,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> Any:
    """Extract a region of interest from a frame with optional padding.

    Parameters
    ----------
    frame : numpy.ndarray
        Source image.
    bbox : tuple
        ``(x1, y1, x2, y2)`` bounding box.
    padding : int
        Pixel padding around the ROI.
    """
    h, w = frame.shape[:2]
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    return frame[y1:y2, x1:x2]


def deskew(frame: Any) -> Any:
    """Deskew a text image for better OCR accuracy.

    Uses minimum area rectangle to estimate skew angle.
    """
    try:
        import cv2
        import numpy as np

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 10:
            return frame

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return frame

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            frame, matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        return frame
