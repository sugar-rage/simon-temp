"""Event topic constants and taxonomy.

All event topics are hierarchical dot-separated strings.  Subscribers
can use wildcards via ``event_bus.subscribe("vision.*", handler)``.

Topic Hierarchy::

    system.*            — System lifecycle events
    vision.*            — Vision pipeline events
    speech.*            — Speech subsystem events (FROZEN)
    navigation.*        — Navigation events
    safety.*            — Safety engine events
    logic.*             — Logic controller events
    plugin.*            — Plugin lifecycle events
    health.*            — Health monitoring events
"""

from __future__ import annotations


# ── System Events ────────────────────────────────────────────────────

SYSTEM_STATE_CHANGED = "system.state_changed"
"""Published when the SystemStateMachine transitions. Data: {old, new}."""

SYSTEM_STARTUP_COMPLETE = "system.startup_complete"
"""Published when all subsystems are initialized."""

SYSTEM_SHUTDOWN_REQUESTED = "system.shutdown_requested"
"""Published when graceful shutdown begins."""

SYSTEM_CAPABILITY_CHANGED = "system.capability_changed"
"""Published when a capability changes status. Data: {name, status}."""


# ── Vision Events ────────────────────────────────────────────────────

VISION_WORLD_UPDATE = "vision.world_update"
"""Per-frame unified WorldModel from Perception Fusion. Data: WorldModel."""

VISION_ENTITY_ENTERED = "vision.entity_entered"
"""New entity appeared in view. Data: {entity, track_id}."""

VISION_ENTITY_EXITED = "vision.entity_exited"
"""Tracked entity disappeared. Data: {track_id, last_seen}."""

VISION_HAZARD_DETECTED = "vision.hazard_detected"
"""Safety-critical hazard detected. Bypasses rate limiting.
Data: {detection, hazard_level, distance}."""

VISION_TEXT_WARNING = "vision.text_warning"
"""Safety-critical OCR text warning detected.
Data: {text, priority}."""

VISION_FRAME_CAPTURED = "vision.frame_captured"
"""Raw frame captured from camera. Data: {frame_id, timestamp}."""

VISION_CAMERA_STATUS = "vision.camera_status"
"""Camera status change. Data: {connected, device_id}."""

VISION_SAVE_FACE = "vision.save_face"
"""Save a face embedding. Data: {track_id, embedding}."""


# ── Speech Events ────────────────────────────────────────────────────

SPEECH_COMMAND = "speech.command"
"""Voice command recognized. Data: {action, args, confidence}."""

SPEECH_SPEAKING = "speech.speaking"
"""TTS started speaking. Data: {text, priority}."""

SPEECH_SPEAK = "speech.speak"
"""Request TTS to speak. Data: {text, priority}."""

SPEECH_LISTENING = "speech.listening"
"""STT listening started/stopped. Data: {active}."""

SPEECH_SET_CONTEXT = "speech.set_context"
"""Update speech context / expectation state. Data: {expecting_confirmation: bool, expecting_name: bool}."""

SPEECH_PLAYBACK_COMPLETE = "speech.playback_complete"
"""Published when physical audio playback and acoustic drain complete. Data: {text, success, correlation_id, event_key}."""

VISUAL_CONTEXT_UPDATED = "visual.context_updated"
"""Published when visual intelligence memory is updated by Ollama. Data: {entry}."""


# ── Navigation Events ───────────────────────────────────────────────

NAV_POSITION = "navigation.position"
"""GPS position update. Data: {location: GeoLocation}."""

NAV_ROUTE_COMPUTED = "navigation.route_computed"
"""Route computation completed. Data: {route: Route}."""

NAV_INSTRUCTION = "navigation.instruction"
"""Turn-by-turn instruction. Data: {instruction, distance_m, maneuver}."""

NAV_OFF_ROUTE = "navigation.off_route"
"""User has deviated from route. Data: {distance_m, direction}."""

NAV_ARRIVED = "navigation.arrived"
"""User arrived at destination. Data: {destination_name}."""

NAV_GPS_STATUS = "navigation.gps_status"
"""GPS fix status change. Data: {has_fix, accuracy_m}."""


# ── Safety Events ────────────────────────────────────────────────────

SAFETY_HAZARD = "safety.hazard"
"""Hazard evaluation result. Data: {level, message, detections}."""

SAFETY_EMERGENCY = "safety.emergency"
"""Emergency requiring immediate TTS. Data: {message, priority}."""

SAFETY_ALL_CLEAR = "safety.all_clear"
"""No hazards in view (after hazard resolved). Data: {}."""


# ── Logic Events ─────────────────────────────────────────────────────

LOGIC_ANNOUNCE = "logic.announce"
"""Request speech announcement. Data: {text, priority}."""

LOGIC_DECISION = "logic.decision"
"""Logic controller made a decision. Data: {action, reasoning}."""

LOGIC_TASK_COMPLETE = "logic.task_complete"
"""Planned task completed. Data: {task_id, result}."""


# ── Plugin Events ────────────────────────────────────────────────────

PLUGIN_LOADED = "plugin.loaded"
"""Plugin successfully loaded. Data: {name, version}."""

PLUGIN_UNLOADED = "plugin.unloaded"
"""Plugin unloaded. Data: {name}."""

PLUGIN_ERROR = "plugin.error"
"""Plugin encountered an error. Data: {name, error}."""


# ── Health Events ────────────────────────────────────────────────────

HEALTH_CHECK = "health.check"
"""Periodic health check result. Data: {subsystem, status}."""

HEALTH_THREAD_DEAD = "health.thread_dead"
"""Watchdog detected a dead thread. Data: {thread_name, last_heartbeat}."""

HEALTH_RESOURCE_LOW = "health.resource_low"
"""System resource running low. Data: {resource, current, threshold}."""


# ── Priority Mapping ─────────────────────────────────────────────────

# Map topic prefixes to default priorities for automatic priority assignment
TOPIC_DEFAULT_PRIORITY: dict[str, int] = {
    "safety.emergency": 1,
    "safety.hazard": 2,
    "vision.hazard_detected": 2,
    "navigation.instruction": 4,
    "navigation.off_route": 3,
    "vision.world_update": 6,
    "speech.command": 4,
    "logic.announce": 5,
    "health.thread_dead": 2,
    "plugin.": 8,
    "system.": 6,
}
