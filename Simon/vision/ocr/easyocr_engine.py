"""EasyOCR engine — fallback text recognition backend."""

from __future__ import annotations

import logging
from typing import Any

from core.config.system_config import OCRConfig
from vision.ocr.base import BaseOCREngine, OCRResult

logger = logging.getLogger("simon.vision.ocr")


class EasyOCREngine(BaseOCREngine):
    """EasyOCR text recognition engine.

    Used as fallback when pytesseract is unavailable.

    Parameters
    ----------
    config : OCRConfig
        OCR configuration.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self._config = config or OCRConfig()
        self._reader: Any = None
        self._ready = False
        self._init()

    def _init(self) -> None:
        try:
            import easyocr

            self._reader = easyocr.Reader(
                self._config.languages,
                gpu=True,
                verbose=False,
            )
            self._ready = True
            logger.info("EasyOCR engine initialized (languages=%s)", self._config.languages)
        except ImportError:
            logger.warning("easyocr not installed")
        except Exception as e:
            logger.error("EasyOCR init failed: %s", e)

    def read_text(self, frame: Any) -> list[OCRResult]:
        if not self._ready or self._reader is None:
            return []

        try:
            raw_results = self._reader.readtext(frame)
            results: list[OCRResult] = []

            for bbox_points, text, conf in raw_results:
                if conf < self._config.confidence_threshold:
                    continue
                text = text.strip()
                if not text:
                    continue

                # Convert polygon to axis-aligned bbox
                xs = [int(p[0]) for p in bbox_points]
                ys = [int(p[1]) for p in bbox_points]
                bbox = (min(xs), min(ys), max(xs), max(ys))

                results.append(
                    OCRResult(text=text, confidence=conf, bbox=bbox)
                )
            return results

        except Exception as e:
            logger.error("EasyOCR failed: %s", e)
            return []

    def is_ready(self) -> bool:
        return self._ready
