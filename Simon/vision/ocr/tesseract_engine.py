"""Tesseract OCR engine — primary text recognition backend.

Refactored from ``perception/ocr_reader.py`` with config-driven parameters,
preprocessing pipeline, and OCRResult output.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config.system_config import OCRConfig
from vision.ocr.base import BaseOCREngine, OCRResult

logger = logging.getLogger("simon.vision.ocr")


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine via pytesseract.

    Parameters
    ----------
    config : OCRConfig
        OCR configuration.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self._config = config or OCRConfig()
        self._ready = False
        self._pytesseract: Any = None
        self._init()

    def _init(self) -> None:
        try:
            import os
            import pytesseract

            tessdata_dir = os.path.abspath("data/tessdata")
            os.environ["TESSDATA_PREFIX"] = tessdata_dir

            self._pytesseract = pytesseract
            self._ready = True
            logger.info("Tesseract OCR engine initialized")
        except ImportError:
            logger.warning("pytesseract not installed")
            self._ready = False

    def read_text(self, frame: Any) -> list[OCRResult]:
        if not self._ready or self._pytesseract is None:
            return []

        try:
            import cv2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            data = self._pytesseract.image_to_data(
                gray, output_type=self._pytesseract.Output.DICT,
                lang="+".join(self._config.languages),
            )

            results: list[OCRResult] = []
            
            for i, text in enumerate(data["text"]):
                text = text.strip()
                if not text:
                    continue

                conf = float(data["conf"][i]) / 100.0
                
                # Temporarily lowered threshold to 0.1 for debugging
                if conf < 0.1:
                    continue

                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                results.append(
                    OCRResult(
                        text=text,
                        confidence=conf,
                        bbox=(x, y, x + w, y + h),
                    )
                )
            
            return results

        except Exception as e:
            logger.error("Tesseract OCR failed: %s", e)
            return []

    def is_ready(self) -> bool:
        return self._ready
