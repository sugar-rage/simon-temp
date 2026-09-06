"""Vision Renderer — overlays perception data onto camera frames.

This module draws bounding boxes, labels, face identities, and OCR text
on the raw camera frames for debugging and visualization purposes.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from core.models.frame import FrameData
from vision.pipeline.perception_fusion import WorldModel

logger = logging.getLogger("simon.vision.renderer")


class VisionRenderer:
    """Renders vision pipeline results to an OpenCV window.
    
    Parameters
    ----------
    window_name : str
        The name of the OpenCV display window.
    enabled : bool
        Whether rendering is enabled. If False, rendering is bypassed.
    """

    def __init__(self, window_name: str = "SIMON Vision", enabled: bool = True):
        self._window_name = window_name
        self._enabled = enabled
        self._initialized = False
        self._ui_state: dict = {}

    def set_ui_state(self, ui_state: dict) -> None:
        """Update cached UI state from visual information manager."""
        if ui_state:
            self._ui_state.update(ui_state)

    def render(self, frame_data, world: WorldModel, ui_state: Optional[dict] = None) -> None:
        """Draw annotations on the frame and display it.
        
        Parameters
        ----------
        frame_data : numpy.ndarray
            The raw camera frame (BGR format).
        world : WorldModel
            The fully fused world model containing entities and OCR results.
        ui_state : dict, optional
            Visual context and TTS/Ollama status overlay state.
        """
        if not self._enabled or cv2 is None or frame_data is None:
            return
            
        if not self._initialized:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize VisionRenderer window: %s", e)
                self._enabled = False
                return

        if ui_state:
            self._ui_state.update(ui_state)

        # Create a copy so we don't mutate the original frame passed to others
        annotated = frame_data.copy()

        # 1 & 2. Draw YOLO Objects and Faces
        for entity in world.entities:
            x1, y1, x2, y2 = entity.bbox
            
            # Standard object box (e.g. YOLO person box)
            label = f"{entity.cls} {entity.confidence:.2f}"
            color = (255, 0, 0) # Blue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            
            # Draw specific face bounding box if available
            if entity.face_bbox:
                fx1, fy1, fx2, fy2 = entity.face_bbox
                
                # Add 10% padding
                fw = fx2 - fx1
                fh = fy2 - fy1
                pad_w = int(fw * 0.1)
                pad_h = int(fh * 0.1)
                
                fx1 = max(0, fx1 - pad_w)
                fy1 = max(0, fy1 - pad_h)
                fx2 = min(annotated.shape[1], fx2 + pad_w)
                fy2 = min(annotated.shape[0], fy2 + pad_h)
                
                face_label = entity.face_name if entity.face_name else "Unknown"
                
                # Color coding:
                #   Known named   → GREEN  (0, 255, 0)
                #   Anonymous     → BLUE   (255, 128, 0) — "Person 1" etc.
                #   Unknown       → ORANGE (0, 165, 255)
                if face_label == "Unknown":
                    face_color = (0, 165, 255)   # Orange
                elif face_label.startswith("Person "):
                    face_color = (255, 128, 0)   # Blue
                else:
                    face_color = (0, 255, 0)     # Green
                
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), face_color, 2)
                cv2.putText(
                    annotated,
                    face_label,
                    (fx1, max(fy1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    face_color,
                    2,
                )

        # 3. Draw OCR text
        for ocr in world.ocr_texts:
            if ocr.bbox:
                x1, y1, x2, y2 = ocr.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
                cv2.putText(
                    annotated,
                    ocr.text,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # 4. Draw Context-Aware Intelligence HUD Overlay
        self._draw_hud(annotated, world, self._ui_state)

        try:
            cv2.imshow(self._window_name, annotated)
            cv2.waitKey(1)
        except Exception as e:
            logger.error("Failed to render frame: %s", e)

    def _draw_hud(self, annotated, world: WorldModel, ui_state: Optional[dict] = None) -> None:
        """Render a compact, readable context overlay showing SIMON's understanding."""
        if cv2 is None or annotated is None:
            return

        h, w = annotated.shape[:2]
        hud_w = min(380, w - 20)
        hud_h = 240
        x1 = w - hud_w - 10
        y1 = 10
        x2 = w - 10
        y2 = y1 + hud_h

        # Semi-transparent background panel
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 1)
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        state = ui_state or {}

        # 1. OBJECTS
        objs = state.get("objects") or [e.cls for e in world.entities if e.cls]
        obj_str = ", ".join(objs[:3]) if objs else "None"

        # 2. FACES
        faces = state.get("faces") or [e.face_name for e in world.entities if e.face_name]
        face_str = ", ".join(faces[:2]) if faces else "None"

        # 3. OCR
        ocrs = state.get("ocr") or [o.text.strip() for o in world.ocr_texts if o.text and o.text.strip()]
        ocr_str = ", ".join(ocrs[:2]) if ocrs else "None"
        if len(ocr_str) > 28:
            ocr_str = ocr_str[:25] + "..."

        # 4. CLASSIFICATION
        classification = state.get("classification") or "NORMAL"
        clf_color = (0, 0, 255) if "SAFETY" in classification else (
            (0, 255, 255) if "NAV" in classification else (200, 200, 200)
        )

        # 5. DIRECTION & 6. DISTANCE
        first_ent = world.entities[0] if world.entities else None
        direction = state.get("direction") or ((first_ent.position or "Ahead").capitalize() if first_ent else "Ahead")
        distance = state.get("distance") or ((first_ent.distance or "Nearby").capitalize() if first_ent else "Nearby")

        # 7. TTS (current spoken message)
        tts_msg = state.get("tts_text") or "None"
        if len(tts_msg) > 30:
            tts_msg = tts_msg[:27] + "..."

        # 8. TTS STATUS
        tts_status = state.get("tts_status") or ("Speaking" if tts_msg != "None" else "Completed")
        tts_stat_color = (0, 255, 0) if tts_status == "Speaking" else (200, 200, 200)

        # 9. OLLAMA
        ollama_status = state.get("ollama_status") or "Completed"
        if "Waiting" in ollama_status:
            ollama_color = (0, 165, 255)  # Orange
        elif "Processing" in ollama_status:
            ollama_color = (0, 255, 255)  # Yellow
        elif "Unavailable" in ollama_status:
            ollama_color = (128, 128, 128)  # Gray
        else:
            ollama_color = (255, 128, 255)  # Magenta

        lines = [
            ("SIMON CONTEXT INTELLIGENCE", (0, 255, 255), 0.42, 1),
            (f"OBJECTS:        {obj_str}", (255, 255, 255), 0.35, 1),
            (f"FACES:          {face_str}", (0, 255, 128) if face_str != "None" else (180, 180, 180), 0.35, 1),
            (f"OCR:            {ocr_str}", (0, 255, 255) if ocr_str != "None" else (180, 180, 180), 0.35, 1),
            (f"CLASSIFICATION: {classification}", clf_color, 0.35, 1),
            (f"DIRECTION:      {direction}", (255, 255, 255), 0.35, 1),
            (f"DISTANCE:       {distance}", (255, 255, 255), 0.35, 1),
            (f"TTS MESSAGE:    {tts_msg}", (255, 255, 255), 0.35, 1),
            (f"TTS STATUS:     {tts_status}", tts_stat_color, 0.35, 1),
            (f"OLLAMA:         {ollama_status}", ollama_color, 0.35, 1),
        ]

        text_y = y1 + 18
        for text, color, scale, thickness in lines:
            cv2.putText(
                annotated,
                text,
                (x1 + 8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            text_y += 22

    def close(self) -> None:
        """Close the display window safely."""
        if self._initialized and cv2 is not None:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass
            finally:
                self._initialized = False
