"""AppController — the sole orchestration point for SIMON.

This is the composition root: it creates all subsystems, wires
dependencies via constructors, manages the application lifecycle,
and dispatches voice commands through the planner → executor pipeline.

``main.py`` delegates entirely to this class.

Design decisions (see TDR-009):
- ``main.py`` is a ~15-line bootstrap; all logic lives here.
- All dependencies are explicitly wired via constructors (TDR-002).
- Subsystem failures degrade gracefully via CapabilityRegistry (TDR-003).
- System state is tracked via StateMachine (TDR-004).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

from core.config.system_config import SystemConfig
from core.events.event_bus import EventBus
from core.events import event_types
from core.models.events import Event
from core.models.enums import Priority
from core.state.state_machine import StateMachine, SystemState
from core.capabilities.registry import CapabilityRegistry
from core.metrics.collector import SystemMetricsCollector

# Phase 5 components
from core.planner.task_planner import TaskPlanner
from core.planner.task import Task
from core.actions.action_executor import ActionExecutor
from core.actions.action_registry import ActionRegistry
from core.actions.action_handlers import register_builtin_handlers
from core.resources.model_registry import ModelRegistry
from core.resources.resource_manager import ResourceManager
from core.health.watchdog import Watchdog
from core.health.health_monitor import HealthMonitor
from core.plugins.hooks import HookRegistry
from core.plugins.plugin_manager import PluginManager

logger = logging.getLogger("simon.app")


class AppController:
    """Top-level application orchestrator.

    Parameters
    ----------
    config : SystemConfig
        System configuration (loaded from YAML).
    """

    def __init__(self, config: SystemConfig) -> None:
        self._config = config
        self._start_time = time.time()
        self._shutdown_event = threading.Event()

        # ── Core infrastructure (always created) ─────────────────────
        self._event_bus = EventBus(
            rate_limit_s=config.event_bus_rate_limit_s,
            max_queue_size=config.event_bus_queue_size,
        )
        self._capabilities = CapabilityRegistry()
        self._state_machine = StateMachine(initial_state=SystemState.STARTING)
        self._model_registry = ModelRegistry()

        # ── Phase 5 components ───────────────────────────────────────
        self._watchdog = Watchdog(
            self._event_bus,
            interval_s=config.watchdog_interval_s,
            timeout_s=config.watchdog_interval_s * 3,
        )
        self._resource_manager = ResourceManager(
            self._model_registry,
            self._capabilities,
        )
        self._action_registry = ActionRegistry()
        self._hook_registry = HookRegistry()

        # Subscribe to logic announcements for speech
        self._event_bus.subscribe(
            event_types.LOGIC_ANNOUNCE,
            self._handle_logic_announce,
            source="app_controller",
        )
        self._event_bus.subscribe(
            event_types.SPEECH_SET_CONTEXT,
            self._handle_speech_set_context,
            source="app_controller",
        )

        # ── Subsystem references (populated during start) ───────────
        self._speech_manager: Optional[object] = None
        self._logic_controller: Optional[object] = None
        self._safety_engine: Optional[object] = None
        self._vision_pipeline: Optional[object] = None
        self._nav_pipeline: Optional[object] = None
        self._task_planner: Optional[TaskPlanner] = None
        self._action_executor: Optional[ActionExecutor] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._plugin_manager: Optional[PluginManager] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize all subsystems and transition to READY.

        Initialization order:
        1. Event Bus
        2. Watchdog
        3. Speech subsystem (graceful fallback)
        4. Vision subsystem (graceful fallback)
        5. Logic + Safety
        6. Navigation subsystem (graceful fallback)
        7. Task Planner + Action Executor
        8. Health Monitor
        9. Plugin System
        10. Transition to READY or DEGRADED
        """
        logger.info("═══ SIMON Starting ═══")
        self._state_machine.transition(SystemState.INITIALIZING)

        # 1. Start Event Bus
        self._event_bus.start()
        logger.info("Event Bus started")

        # 2. Start Watchdog
        self._watchdog.start()
        self._watchdog.register_thread("main")
        logger.info("Watchdog started")

        # 3. Speech subsystem
        self._init_speech()

        # 4. Vision subsystem
        self._init_vision()

        # 5. Logic + Safety
        self._init_logic()

        # 6. Navigation subsystem
        self._init_navigation()

        # 7. Task Planner + Action Executor
        self._init_planner_executor()

        # 8. Health Monitor
        self._health_monitor = HealthMonitor(
            capabilities=self._capabilities,
            watchdog=self._watchdog,
            model_registry=self._model_registry,
            state_machine=self._state_machine,
            event_bus=self._event_bus,
            start_time=self._start_time,
        )
        logger.info("Health Monitor ready")

        # 9. Plugin System
        self._init_plugins()

        # 10. Transition to READY or DEGRADED
        unavailable = self._capabilities.list_unavailable()
        if unavailable:
            self._state_machine.transition(SystemState.DEGRADED)
            logger.warning(
                "System starting in DEGRADED mode. Unavailable: %s", unavailable,
            )
        else:
            self._state_machine.transition(SystemState.READY)

        # Publish startup event
        self._event_bus.publish(Event(
            topic=event_types.SYSTEM_STARTUP_COMPLETE,
            priority=Priority.INFORMATIONAL,
            source="app_controller",
            data={"state": self._state_machine.state.name},
        ))

        # Announce readiness
        state_name = self._state_machine.state.name.lower()
        self._speak(f"SIMON is {state_name}.")

        logger.info(
            "═══ SIMON %s ═══ (%d capabilities registered)",
            self._state_machine.state.name,
            len(self._capabilities.list_all()),
        )

    def run(self) -> None:
        """Main loop — process commands until shutdown.

        Polls for voice commands, dispatches them through the
        planner → executor pipeline, and sends heartbeats to
        the watchdog.
        """
        logger.info("Entering main loop")
        try:
            while not self._shutdown_event.is_set():
                # Heartbeat
                self._watchdog.heartbeat("main")

                # Poll for voice commands
                command = self._poll_command()
                if command is not None:
                    self._handle_command(command)

                # Check for shutdown state
                if self._state_machine.state in (
                    SystemState.SHUTTING_DOWN, SystemState.STOPPED,
                ):
                    break

                # Yield to other threads
                self._shutdown_event.wait(timeout=0.05)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            logger.info("Main loop exited")

    def stop(self) -> None:
        """Graceful shutdown — stop all subsystems in reverse order."""
        if self._shutdown_event.is_set():
            return  # Already shutting down

        logger.info("═══ SIMON Shutting Down ═══")
        self._shutdown_event.set()

        # Transition state
        try:
            if self._state_machine.state != SystemState.SHUTTING_DOWN:
                self._state_machine.transition(SystemState.SHUTTING_DOWN)
        except Exception:
            pass  # May fail if state doesn't allow this transition

        # Publish shutdown event
        try:
            self._event_bus.publish(Event(
                topic=event_types.SYSTEM_SHUTDOWN_REQUESTED,
                priority=Priority.INFORMATIONAL,
                source="app_controller",
            ))
        except Exception:
            pass

        # Cancel active tasks
        if self._task_planner:
            self._task_planner.cancel_all()

        # Unload plugins
        if self._plugin_manager:
            self._plugin_manager.unload_all()

        # Stop logic controller
        if self._logic_controller and hasattr(self._logic_controller, "stop"):
            try:
                self._logic_controller.stop()
            except Exception:
                logger.error("LogicController stop failed", exc_info=True)

        # Stop safety engine (if it has a stop method)
        if self._safety_engine and hasattr(self._safety_engine, "stop"):
            try:
                self._safety_engine.stop()
            except Exception:
                logger.error("SafetyEngine stop failed", exc_info=True)

        # Stop navigation
        if self._nav_pipeline and hasattr(self._nav_pipeline, "stop"):
            try:
                self._nav_pipeline.stop()
            except Exception:
                logger.error("Navigation stop failed", exc_info=True)

        # Stop vision
        if self._vision_pipeline and hasattr(self._vision_pipeline, "stop"):
            try:
                self._vision_pipeline.stop()
            except Exception:
                logger.error("Vision stop failed", exc_info=True)

        # Stop speech
        if self._speech_manager and hasattr(self._speech_manager, "stop"):
            try:
                self._speech_manager.stop()
            except Exception:
                logger.error("SpeechManager stop failed", exc_info=True)

        # Stop watchdog
        self._watchdog.stop()

        # Stop event bus
        self._event_bus.stop()

        # Final transition
        try:
            self._state_machine.transition(SystemState.STOPPED)
        except Exception:
            pass

        logger.info("═══ SIMON Stopped ═══")

    # ── Command handling ─────────────────────────────────────────────

    def _handle_command(self, command: dict) -> None:
        """Process a voice command through planner → executor.

        Parameters
        ----------
        command : dict
            Voice command with ``action`` and optional ``args``.
        """
        action = command.get("action", "")
        args = command.get("args", "")
        raw_transcript = command.get("raw_transcript", "")

        logger.info("Command received: action=%r args=%r", action, args)
        logger.info("[NAME_TRACE] AppController received command: action=%r, args=%r, raw=%r", action, args, raw_transcript)
        logger.info("[CONFIRM_TRACE] AppController received command: action=%r, args=%r, raw=%r", action, args, raw_transcript)

        if not action:
            return

        # Publish to EventBus so LogicController can intercept
        # registration-related responses (yes/no/name)
        logger.info(
            "[EVENT] SPEECH_COMMAND published: action=%r, args=%r, raw=%r",
            action, args, raw_transcript,
        )
        logger.info("[NAME_TRACE] AppController publishing SPEECH_COMMAND: action=%r, args=%r, raw=%r", action, args, raw_transcript)
        logger.info("[CONFIRM_TRACE] AppController publishing SPEECH_COMMAND: action=%r, args=%r, raw=%r", action, args, raw_transcript)
        self._event_bus.publish(Event(
            topic=event_types.SPEECH_COMMAND,
            data={
                "action": action,
                "args": args,
                "raw_transcript": raw_transcript,
            },
            priority=Priority.INFORMATIONAL,
            source="app_controller",
        ))

        # Registration-only actions should not go through the planner
        if action in ("yes", "no", "no_name", "unmatched_text"):
            return

        # Plan
        if self._task_planner is None:
            return
        task = self._task_planner.plan_command(action, args)
        if task is None:
            self._speak(f"I don't know how to {action}.")
            return

        # Execute
        if self._action_executor is None:
            return
        result = self._action_executor.execute_task(task)

        # Update task planner
        if result.success:
            self._task_planner.on_task_complete(task.task_id, result.data)
        else:
            self._task_planner.on_task_failed(
                task.task_id, result.error or "Unknown error",
            )

    def _poll_command(self) -> Optional[dict]:
        """Poll the speech subsystem for a voice command.

        Returns
        -------
        dict or None
            Command dict ``{"action": str, "args": str}`` or None.
        """
        if self._speech_manager is None:
            return None
        try:
            if hasattr(self._speech_manager, "get_command"):
                return self._speech_manager.get_command()
        except Exception:
            logger.error("Speech command poll failed", exc_info=True)
        return None

    def _speak(
        self,
        text: str,
        priority: int = Priority.INFORMATIONAL,
        on_complete: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Convenience: speak text if speech is available."""
        if self._speech_manager and hasattr(self._speech_manager, "speak"):
            try:
                self._speech_manager.speak(text, priority=priority, on_complete=on_complete)
            except Exception:
                logger.error("TTS failed: %s", text, exc_info=True)
                if on_complete:
                    try:
                        on_complete(False)
                    except Exception:
                        pass
        elif on_complete:
            try:
                on_complete(True)
            except Exception:
                pass

    def _handle_logic_announce(self, event: Event) -> None:
        """Handle LOGIC_ANNOUNCE events by speaking them."""
        text = event.data.get("text")
        if text:
            priority = event.data.get("priority", Priority.INFORMATIONAL)
            logger.info("[SPEECH] LOGIC_ANNOUNCE received: %r (priority=%s)", text, priority)
            if "Would you like me to save this person?" in text:
                logger.info("[CONFIRM_TRACE] AppController detected save question, setting expecting_confirmation=True")
                if self._speech_manager and hasattr(self._speech_manager, "set_expecting_confirmation"):
                    self._speech_manager.set_expecting_confirmation(True)
            elif "What should I call this person?" in text:
                logger.info("[CONFIRM_TRACE] AppController detected name prompt, setting expecting_confirmation=False, expecting_name=True")
                if self._speech_manager and hasattr(self._speech_manager, "set_expecting_confirmation"):
                    self._speech_manager.set_expecting_confirmation(False)
                if self._speech_manager and hasattr(self._speech_manager, "set_expecting_name"):
                    self._speech_manager.set_expecting_name(True)
            elif "saved." in text:
                logger.info("[NAME_TRACE] AppController detected 'saved.', resetting speech context")
                if self._speech_manager and hasattr(self._speech_manager, "set_expecting_confirmation"):
                    self._speech_manager.set_expecting_confirmation(False)
                if self._speech_manager and hasattr(self._speech_manager, "set_expecting_name"):
                    self._speech_manager.set_expecting_name(False)

            event_key = event.data.get("event_key")

            def _on_speech_finished(success: bool) -> None:
                self._event_bus.publish(Event(
                    topic=event_types.SPEECH_PLAYBACK_COMPLETE,
                    data={
                        "text": text,
                        "success": success,
                        "event_key": event_key,
                        "correlation_id": event.correlation_id,
                    },
                    priority=Priority.INFORMATIONAL,
                    source="speech_pipeline",
                    correlation_id=event.correlation_id,
                ))

            self._speak(text, priority=priority, on_complete=_on_speech_finished)

    def _handle_speech_set_context(self, event: Event) -> None:
        """Handle SPEECH_SET_CONTEXT events to adjust speech thresholds/state."""
        if "expecting_confirmation" in event.data:
            exp_confirm = event.data.get("expecting_confirmation", False)
            logger.info("[CONFIRM_TRACE] AppController received SPEECH_SET_CONTEXT: expecting_confirmation=%s", exp_confirm)
            if self._speech_manager and hasattr(self._speech_manager, "set_expecting_confirmation"):
                self._speech_manager.set_expecting_confirmation(exp_confirm)
        if "expecting_name" in event.data:
            exp_name = event.data.get("expecting_name", False)
            logger.info("[NAME_TRACE] AppController received SPEECH_SET_CONTEXT: expecting_name=%s", exp_name)
            if self._speech_manager and hasattr(self._speech_manager, "set_expecting_name"):
                self._speech_manager.set_expecting_name(exp_name)

    # ── Subsystem initialization ─────────────────────────────────────

    def _init_speech(self) -> None:
        """Initialize the Speech subsystem (graceful fallback)."""
        try:
            from speech.manager.speech_manager import SpeechManager
            self._speech_manager = SpeechManager()
            if hasattr(self._speech_manager, "start"):
                self._speech_manager.start()
            self._capabilities.register("capability.speech", available=True)
            logger.info("Speech subsystem initialized")
        except ImportError:
            self._capabilities.register(
                "capability.speech", available=False,
                reason="speech module not available",
            )
            logger.warning("Speech subsystem unavailable (import failed)")
        except Exception as exc:
            self._capabilities.register(
                "capability.speech", available=False,
                reason=str(exc),
            )
            logger.warning("Speech subsystem failed to start: %s", exc)

    def _init_vision(self) -> None:
        """Initialize the Vision subsystem (graceful fallback)."""
        try:
            from vision.pipeline.vision_pipeline import VisionPipeline
            from vision.camera.cv2_camera import CV2Camera
            from vision.detection.yolo_detector import YOLODetector
            from vision.ocr.tesseract_engine import TesseractEngine
            from vision.face.insightface_recognizer import InsightFaceRecognizer
            
            camera = CV2Camera(config=self._config.vision.camera)
            detector = YOLODetector(config=self._config.vision.detection)
            ocr_engine = TesseractEngine(config=self._config.vision.ocr)
            face_recognizer = InsightFaceRecognizer(config=self._config.vision.face)
            
            # Load models
            try:
                detector.load()
            except Exception as e:
                logger.warning("Failed to load YOLO detector: %s", e)
                
            try:
                face_recognizer.load()
            except Exception as e:
                logger.warning("Failed to load Face recognizer: %s", e)

            self._vision_pipeline = VisionPipeline(
                camera=camera,
                detector=detector,
                ocr_engine=ocr_engine,
                face_recognizer=face_recognizer,
                config=self._config.vision,
                event_bus=self._event_bus,
                capabilities=self._capabilities,
            )
            if hasattr(self._vision_pipeline, "start"):
                self._vision_pipeline.start()
            logger.info("Vision subsystem initialized")
        except ImportError:
            self._capabilities.register(
                "capability.camera", available=False,
                reason="vision module not available",
            )
            self._capabilities.register(
                "capability.detection", available=False,
                reason="vision module not available",
            )
            logger.warning("Vision subsystem unavailable (import failed)")
        except Exception as exc:
            self._capabilities.register(
                "capability.camera", available=False, reason=str(exc),
            )
            self._capabilities.register(
                "capability.detection", available=False, reason=str(exc),
            )
            logger.warning("Vision subsystem failed to start: %s", exc)

    def _init_logic(self) -> None:
        """Initialize Logic Controller and Safety Engine."""
        try:
            from core.logic.logic_controller import LogicController
            from core.logic.known_person_tracker import KnownPersonTracker
            known_person_tracker = KnownPersonTracker(
                reannounce_timeout_s=getattr(self._config.vision.face, "known_person_reannounce_timeout_s", 300.0)
            )
            self._logic_controller = LogicController(
                event_bus=self._event_bus,
                vision_pipeline=self._vision_pipeline,
                known_person_tracker=known_person_tracker,
            )
            self._logic_controller.start()
            logger.info("Logic Controller started")
        except Exception as exc:
            logger.warning("Logic Controller failed: %s", exc)

        try:
            from core.safety.safety_engine import SafetyEngine
            self._safety_engine = SafetyEngine(
                event_bus=self._event_bus,
                config=self._config.safety,
            )
            if hasattr(self._safety_engine, "start"):
                self._safety_engine.start()
            logger.info("Safety Engine started")
        except Exception as exc:
            logger.warning("Safety Engine failed: %s", exc)

    def _init_navigation(self) -> None:
        """Initialize the Navigation subsystem (graceful fallback)."""
        try:
            from navigation.pipeline import NavigationPipeline
            from navigation.gps.providers import SimulationGPS
            from navigation.routing.routers import OfflineRouter
            self._nav_pipeline = NavigationPipeline(
                gps=SimulationGPS(),
                router=OfflineRouter(),
                config=self._config.navigation,
                event_bus=self._event_bus,
                capabilities=self._capabilities,
            )
            if hasattr(self._nav_pipeline, "start"):
                self._nav_pipeline.start()
            self._capabilities.register("capability.gps", available=True)
            self._capabilities.register("capability.navigation", available=True)
            logger.info("Navigation subsystem initialized")
        except ImportError:
            self._capabilities.register(
                "capability.gps", available=False,
                reason="navigation module not available",
            )
            logger.warning("Navigation subsystem unavailable (import failed)")
        except Exception as exc:
            self._capabilities.register(
                "capability.gps", available=False, reason=str(exc),
            )
            logger.warning("Navigation subsystem failed: %s", exc)

    def _init_planner_executor(self) -> None:
        """Initialize Task Planner and Action Executor."""
        self._task_planner = TaskPlanner(event_bus=self._event_bus)
        self._action_executor = ActionExecutor(
            action_registry=self._action_registry,
            event_bus=self._event_bus,
        )

        # Wire built-in action handlers
        register_builtin_handlers(
            registry=self._action_registry,
            speech_manager=self._speech_manager,
            nav_pipeline=self._nav_pipeline,
            capabilities=self._capabilities,
            health_monitor=self._health_monitor,  # None at this point — set later
            state_machine=self._state_machine,
        )
        logger.info("Task Planner and Action Executor ready")

    def _init_plugins(self) -> None:
        """Initialize the Plugin System."""
        try:
            self._plugin_manager = PluginManager(
                event_bus=self._event_bus,
                config=self._config,
                capabilities=self._capabilities,
                action_registry=self._action_registry,
                hook_registry=self._hook_registry,
                plugin_dir=self._config.plugins.plugin_dir,
            )
            if self._config.plugins.enabled and self._config.plugins.auto_discover:
                loaded = self._plugin_manager.load_all()
                logger.info("Plugin system: %d plugins loaded", loaded)
            else:
                logger.info("Plugin system initialized (discovery disabled)")
            self._capabilities.register("capability.plugins", available=True)
        except Exception as exc:
            self._capabilities.register(
                "capability.plugins", available=False, reason=str(exc),
            )
            logger.warning("Plugin system failed: %s", exc)

    # ── Properties for testing ───────────────────────────────────────

    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state_machine.state

    @property
    def event_bus(self) -> EventBus:
        """Event bus instance."""
        return self._event_bus

    @property
    def capabilities(self) -> CapabilityRegistry:
        """Capability registry instance."""
        return self._capabilities

    @property
    def is_running(self) -> bool:
        """True if the system is in an active state."""
        return self._state_machine.state in (
            SystemState.READY,
            SystemState.LISTENING,
            SystemState.PROCESSING,
            SystemState.NAVIGATING,
            SystemState.DEGRADED,
        )
