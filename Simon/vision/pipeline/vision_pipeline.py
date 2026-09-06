"""Vision pipeline — orchestrates camera → detection → OCR → face → fusion.

This is the top-level vision component. It owns the camera manager
and runs periodic detection, OCR, face recognition, and scene analysis
on captured frames, then fuses results through PerceptionFusion.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from core.models.frame import FrameData
from core.events.event_bus import EventBus
from core.config.system_config import VisionConfig
from core.capabilities.registry import CapabilityRegistry
from core.metrics.collector import SystemMetricsCollector
from vision.camera.base import BaseCameraSource
from vision.camera.camera_manager import CameraManager
from vision.detection.base import BaseDetector
from vision.detection.tracker import DetectionTracker
from vision.ocr.base import BaseOCREngine
from vision.face.base import BaseFaceRecognizer
from vision.scene.scene_analyzer import SceneAnalyzer
from vision.pipeline.perception_fusion import PerceptionFusion, WorldModel
from vision.vision_renderer import VisionRenderer

logger = logging.getLogger("simon.vision.pipeline")


class VisionPipeline:
    """Orchestrates the complete vision processing pipeline.

    Parameters
    ----------
    camera : BaseCameraSource
        Camera source (injected).
    detector : BaseDetector
        Object detector (injected).
    event_bus : EventBus
        Event bus for publishing results.
    config : VisionConfig
        Vision configuration.
    ocr_engine : BaseOCREngine, optional
        OCR engine (optional capability).
    face_recognizer : BaseFaceRecognizer, optional
        Face recognizer (optional capability).
    capabilities : CapabilityRegistry, optional
        For registering vision capabilities.
    """

    def __init__(
        self,
        camera: BaseCameraSource,
        detector: BaseDetector,
        event_bus: EventBus,
        config: Optional[VisionConfig] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        face_recognizer: Optional[BaseFaceRecognizer] = None,
        capabilities: Optional[CapabilityRegistry] = None,
    ) -> None:
        self._config = config or VisionConfig()
        self._event_bus = event_bus
        self._capabilities = capabilities

        # Camera
        self._camera_manager = CameraManager(
            camera, config=self._config.camera
        )

        # Vision components
        self._detector = detector
        self._ocr_engine = ocr_engine
        self._face_recognizer = face_recognizer
        self._scene_analyzer = SceneAnalyzer()
        self._renderer = VisionRenderer()

        # Perception fusion
        self._tracker = DetectionTracker()
        self._fusion = PerceptionFusion(
            tracker=self._tracker,
            frame_width=self._config.camera.width,
        )

        # State
        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._latest_world: Optional[WorldModel] = None
        self._metrics = SystemMetricsCollector.get()

    @property
    def latest_world(self) -> Optional[WorldModel]:
        """Return the most recent WorldModel."""
        return self._latest_world

    def start(self) -> bool:
        """Start the camera and vision processing pipeline."""
        if self._running:
            return True

        # Register capabilities
        if self._capabilities:
            self._capabilities.register(
                "capability.detection",
                available=self._detector.is_ready(),
            )
            if self._ocr_engine:
                self._capabilities.register(
                    "capability.ocr",
                    available=self._ocr_engine.is_ready(),
                )
            if self._face_recognizer:
                self._capabilities.register(
                    "capability.face_recognition",
                    available=self._face_recognizer.is_ready(),
                )

        if not self._camera_manager.start():
            if self._capabilities:
                self._capabilities.register(
                    "capability.camera", available=False, reason="failed to open"
                )
            return False

        if self._capabilities:
            self._capabilities.register("capability.camera", available=True)

        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="VisionPipeline",
            daemon=True,
        )
        self._process_thread.start()
        
        if self._face_recognizer:
            from core.events import event_types
            self._event_bus.subscribe(
                event_types.VISION_SAVE_FACE,
                self._on_save_face,
                source="vision_pipeline",
            )
            
        logger.info("Vision pipeline started")
        return True

    def _on_save_face(self, event) -> None:
        """Handle request to save a recognized face."""
        if not self._face_recognizer:
            return
            
        embedding = event.data.get("embedding")
        name = event.data.get("name")
        track_id = event.data.get("track_id")
        logger.info(
            "[NAME_TRACE] VisionPipeline _on_save_face received: name=%r, track_id=%s, embedding_present=%s",
            name, track_id, (embedding is not None),
        )
        if embedding is None or not name:
            return
        
        from vision.face.base import FaceResult
        face = FaceResult(
            name="Unknown",
            confidence=1.0,
            bbox=(0, 0, 0, 0),
            embedding=embedding,
        )
        
        import os
        save_dir = os.path.join(getattr(self._face_recognizer, "_db_dir", "data/faces"), name)
        logger.info("[NAME_TRACE] save path: %s", save_dir)
        
        # InsightFaceRecognizer only needs the embedding to save it to the DB
        success = self._face_recognizer.save_face(None, face, name)
        if success:
            logger.info(f"[REGISTRATION] Person saved successfully as {name!r}")
            logger.info(f"[NAME_TRACE] Person saved successfully as {name!r}")
        else:
            logger.error(f"[REGISTRATION] Failed to save face {name}")
            logger.error(f"[NAME_TRACE] Failed to save face {name}")

    def stop(self) -> None:
        """Stop the vision pipeline."""
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=5.0)
        self._camera_manager.stop()
        self._renderer.close()
        logger.info("Vision pipeline stopped")

    def _process_loop(self) -> None:
        """Main processing loop — runs on dedicated thread."""
        while self._running:
            frame = self._camera_manager.get_frame()
            if frame is None or not frame.is_valid:
                time.sleep(0.01)
                continue

            self._frame_count += 1
            self._process_frame(frame)

    def _process_frame(self, frame: FrameData) -> None:
        """Process a single frame through the full pipeline."""
        with self._metrics.timer("vision.frame_latency_ms"):
            # Object detection (every frame)
            detections = self._detector.detect(frame.data)
            self._metrics.increment("vision.frame_count")

            # Face recognition (periodic)
            faces = None
            if (
                self._face_recognizer
                and self._face_recognizer.is_ready()
                and self._frame_count % self._config.face.detection_interval_frames == 0
            ):
                faces = self._face_recognizer.detect_faces(frame.data)

            # OCR (periodic)
            ocr_results = None
            if (
                self._ocr_engine
                and self._ocr_engine.is_ready()
                and self._frame_count % self._config.ocr_interval_frames == 0
            ):
                ocr_results = self._ocr_engine.read_text(frame.data)

            # Scene analysis (periodic)
            scene = None
            if self._frame_count % self._config.scene_analysis_interval_frames == 0:
                scene = self._scene_analyzer.analyze(detections)

            # Fuse into WorldModel
            world = self._fusion.fuse(
                detections=detections,
                faces=faces,
                ocr_results=ocr_results,
                scene=scene,
                frame_id=frame.frame_id,
            )
            self._latest_world = world

            # Generate and publish events
            events = self._fusion.get_events(world)
            for event in events:
                self._event_bus.publish(event)
                
            # Render frame
            self._renderer.render(frame.data, world)
