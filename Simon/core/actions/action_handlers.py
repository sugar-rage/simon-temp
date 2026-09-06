"""Built-in action handlers for standard SIMON voice commands.

Each handler is a function that takes an ``Action`` plus the specific
subsystem references it needs, and returns an ``ActionResult``.

The ``register_builtin_handlers`` convenience function wires all
handlers into an ``ActionRegistry``, binding subsystem references
via closures.

Design decision: handlers are plain functions (not methods) to keep
them stateless and independently testable.  Subsystem dependencies
are bound at registration time via ``functools.partial``-style closures.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from core.models.actions import Action, ActionResult
from core.models.enums import Priority
from core.actions.action_registry import ActionRegistry

logger = logging.getLogger("simon.core.actions")


def handle_speak(action: Action, speech_manager: Any) -> ActionResult:
    """Speak text via TTS.

    Expected args: ``{"text": str, "priority": int}``.
    """
    t0 = time.time()
    text = action.args.get("text", "")
    priority = action.args.get("priority", Priority.INFORMATIONAL)
    if not text:
        return ActionResult(
            action=action, success=False, message="No text to speak",
            duration_ms=(time.time() - t0) * 1000,
        )
    try:
        speech_manager.speak(text, priority=priority)
        return ActionResult(
            action=action, success=True, message=f"Speaking: {text[:50]}",
            duration_ms=(time.time() - t0) * 1000,
        )
    except Exception as exc:
        return ActionResult(
            action=action, success=False, error=str(exc),
            duration_ms=(time.time() - t0) * 1000,
        )


def handle_navigate(action: Action, speech_manager: Any, nav_pipeline: Any) -> ActionResult:
    """Start navigation to a destination.

    Expected args: ``{"destination": str}`` or ``{"args": str}``.
    """
    t0 = time.time()
    dest = action.args.get("destination", action.args.get("args", ""))
    if not dest:
        speech_manager.speak("Where would you like to go?", priority=Priority.INFORMATIONAL)
        return ActionResult(
            action=action, success=False, message="No destination provided",
            duration_ms=(time.time() - t0) * 1000,
        )
    try:
        nav_pipeline.navigate_to(dest)
        return ActionResult(
            action=action, success=True, message=f"Navigating to {dest}",
            duration_ms=(time.time() - t0) * 1000,
        )
    except Exception as exc:
        speech_manager.speak("Navigation failed. Please try again.", priority=Priority.INFORMATIONAL)
        return ActionResult(
            action=action, success=False, error=str(exc),
            duration_ms=(time.time() - t0) * 1000,
        )


def handle_cancel_nav(action: Action, speech_manager: Any, nav_pipeline: Any) -> ActionResult:
    """Cancel active navigation."""
    t0 = time.time()
    try:
        nav_pipeline.cancel()
        speech_manager.speak("Navigation cancelled.", priority=Priority.INFORMATIONAL)
        return ActionResult(
            action=action, success=True, message="Navigation cancelled",
            duration_ms=(time.time() - t0) * 1000,
        )
    except Exception as exc:
        return ActionResult(
            action=action, success=False, error=str(exc),
            duration_ms=(time.time() - t0) * 1000,
        )


def handle_read_text(action: Action, speech_manager: Any, capabilities: Any) -> ActionResult:
    """Read text from the current camera view (via OCR).

    Design note: the actual OCR read is triggered via the vision pipeline,
    not directly here.  This handler announces the request; the vision
    pipeline publishes OCR results through the event bus.
    """
    t0 = time.time()
    if not capabilities.is_available("capability.ocr"):
        speech_manager.speak(
            "Text reading is not available.", priority=Priority.INFORMATIONAL,
        )
        return ActionResult(
            action=action, success=False, message="OCR not available",
            duration_ms=(time.time() - t0) * 1000,
        )
    speech_manager.speak("Reading text...", priority=Priority.INFORMATIONAL)
    return ActionResult(
        action=action, success=True, message="OCR triggered",
        duration_ms=(time.time() - t0) * 1000,
    )


def handle_save_face(action: Action, speech_manager: Any, capabilities: Any) -> ActionResult:
    """Save an unknown face with a name."""
    t0 = time.time()
    name = action.args.get("args", action.args.get("name", ""))
    if not capabilities.is_available("capability.face_recognition"):
        speech_manager.speak(
            "Face recognition is not available.", priority=Priority.INFORMATIONAL,
        )
        return ActionResult(
            action=action, success=False, message="Face recognition not available",
            duration_ms=(time.time() - t0) * 1000,
        )
    if not name:
        speech_manager.speak(
            "Please say the name after save face.", priority=Priority.INFORMATIONAL,
        )
        return ActionResult(
            action=action, success=False, message="No name provided",
            duration_ms=(time.time() - t0) * 1000,
        )
    speech_manager.speak(f"Saving face as {name}.", priority=Priority.INFORMATIONAL)
    return ActionResult(
        action=action, success=True, message=f"Face save triggered for {name}",
        duration_ms=(time.time() - t0) * 1000,
    )


def handle_describe_scene(action: Action, speech_manager: Any, capabilities: Any) -> ActionResult:
    """Describe the current scene."""
    t0 = time.time()
    if not capabilities.is_available("capability.camera"):
        speech_manager.speak("Camera is not available.", priority=Priority.INFORMATIONAL)
        return ActionResult(
            action=action, success=False, message="Camera not available",
            duration_ms=(time.time() - t0) * 1000,
        )
    speech_manager.speak("Describing scene...", priority=Priority.INFORMATIONAL)
    return ActionResult(
        action=action, success=True, message="Scene description triggered",
        duration_ms=(time.time() - t0) * 1000,
    )


def handle_status(
    action: Action, speech_manager: Any, capabilities: Any, health_monitor: Any,
) -> ActionResult:
    """Report system status via TTS."""
    t0 = time.time()
    try:
        report = health_monitor.get_status_report()
        speech_manager.speak(report, priority=Priority.LOW)
        return ActionResult(
            action=action, success=True, message="Status reported",
            data={"report": report},
            duration_ms=(time.time() - t0) * 1000,
        )
    except Exception as exc:
        speech_manager.speak("Unable to get status.", priority=Priority.INFORMATIONAL)
        return ActionResult(
            action=action, success=False, error=str(exc),
            duration_ms=(time.time() - t0) * 1000,
        )


def handle_stop(action: Action, speech_manager: Any, state_machine: Any) -> ActionResult:
    """Initiate graceful shutdown."""
    t0 = time.time()
    speech_manager.speak("Goodbye.", priority=Priority.INFORMATIONAL)
    try:
        from core.state.state_machine import SystemState
        state_machine.transition(SystemState.SHUTTING_DOWN)
    except Exception:
        pass  # Shutdown will happen in the main loop
    return ActionResult(
        action=action, success=True, message="Shutdown initiated",
        duration_ms=(time.time() - t0) * 1000,
    )


def register_builtin_handlers(
    registry: ActionRegistry,
    speech_manager: Any,
    nav_pipeline: Any = None,
    capabilities: Any = None,
    health_monitor: Any = None,
    state_machine: Any = None,
) -> None:
    """Wire all built-in handlers into the registry.

    Subsystem references are bound via closures so each handler
    becomes a simple ``(Action) -> ActionResult`` callable.

    Parameters
    ----------
    registry : ActionRegistry
        Target registry to populate.
    speech_manager : SpeechManager
        TTS/STT facade.
    nav_pipeline : NavigationPipeline, optional
        Navigation subsystem.
    capabilities : CapabilityRegistry, optional
        Runtime capability checks.
    health_monitor : HealthMonitor, optional
        For status reports.
    state_machine : StateMachine, optional
        For shutdown transitions.
    """
    # Speak
    registry.register("speak", lambda a: handle_speak(a, speech_manager))

    # Navigation
    if nav_pipeline is not None:
        registry.register(
            "navigate",
            lambda a: handle_navigate(a, speech_manager, nav_pipeline),
        )
        registry.register(
            "cancel_nav",
            lambda a: handle_cancel_nav(a, speech_manager, nav_pipeline),
        )

    # Vision-triggered
    if capabilities is not None:
        registry.register(
            "read_text",
            lambda a: handle_read_text(a, speech_manager, capabilities),
        )
        registry.register(
            "save_face",
            lambda a: handle_save_face(a, speech_manager, capabilities),
        )
        registry.register(
            "describe_scene",
            lambda a: handle_describe_scene(a, speech_manager, capabilities),
        )

    # Status
    if health_monitor is not None and capabilities is not None:
        registry.register(
            "status",
            lambda a: handle_status(a, speech_manager, capabilities, health_monitor),
        )

    # Stop
    if state_machine is not None:
        registry.register(
            "stop",
            lambda a: handle_stop(a, speech_manager, state_machine),
        )
