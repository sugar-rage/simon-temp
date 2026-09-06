"""
Message-type to speech style mapping for Emotional TTS.

Ensures that urgency is communicated through vocal parameters,
not just words, providing acoustic cues to visually impaired users.
"""

from __future__ import annotations

from dataclasses import dataclass

from speech.models.speech_priority import SpeechPriority


@dataclass(frozen=True)
class EmotionalStyle:
    """Acoustic profile for speech synthesis."""
    speed_factor: float
    pitch_shift_pct: int
    volume_pct: int
    emphasis_level: str
    description: str


# ---------------------------------------------------------------------- #
#  Style Definitions
# ---------------------------------------------------------------------- #

# Extreme urgency, clipping, loud
_EMERGENCY_STYLE = EmotionalStyle(
    speed_factor=1.3,
    pitch_shift_pct=20,
    volume_pct=100,
    emphasis_level="Strong",
    description="Urgent, clipped",
)

# High urgency, clear
_OBSTACLE_STYLE = EmotionalStyle(
    speed_factor=1.2,
    pitch_shift_pct=10,
    volume_pct=95,
    emphasis_level="Moderate",
    description="Alert, clear",
)

# Standard pace, steady
_NAVIGATION_STYLE = EmotionalStyle(
    speed_factor=1.0,
    pitch_shift_pct=0,
    volume_pct=85,
    emphasis_level="Calm",
    description="Steady, measured",
)

# Slower, highly articulated
_OCR_STYLE = EmotionalStyle(
    speed_factor=0.85,
    pitch_shift_pct=0,
    volume_pct=80,
    emphasis_level="Clear",
    description="Articulated",
)

# Friendly, normal pace
_FACE_STYLE = EmotionalStyle(
    speed_factor=1.0,
    pitch_shift_pct=0,
    volume_pct=80,
    emphasis_level="Warm",
    description="Friendly",
)

# Informational, neutral
_STATUS_STYLE = EmotionalStyle(
    speed_factor=1.0,
    pitch_shift_pct=0,
    volume_pct=75,
    emphasis_level="Neutral",
    description="Informational",
)

# Slightly slower, lower pitch, questioning
_CONFIRMATION_STYLE = EmotionalStyle(
    speed_factor=0.95,
    pitch_shift_pct=-5,
    volume_pct=80,
    emphasis_level="Gentle",
    description="Questioning",
)

# Faster, slightly higher pitch, concerned
_ERROR_STYLE = EmotionalStyle(
    speed_factor=1.1,
    pitch_shift_pct=5,
    volume_pct=90,
    emphasis_level="Moderate",
    description="Concerned",
)


def get_style_for_priority(priority: SpeechPriority) -> EmotionalStyle:
    """Map a SpeechPriority to its corresponding EmotionalStyle."""
    mapping = {
        SpeechPriority.EMERGENCY: _EMERGENCY_STYLE,
        SpeechPriority.OBSTACLE: _OBSTACLE_STYLE,
        SpeechPriority.NAVIGATION: _NAVIGATION_STYLE,
        SpeechPriority.OCR: _OCR_STYLE,
        SpeechPriority.FACE: _FACE_STYLE,
        SpeechPriority.STATUS: _STATUS_STYLE,
        SpeechPriority.NOTIFICATION: _STATUS_STYLE,
        SpeechPriority.INFO: _STATUS_STYLE,
        SpeechPriority.CONFIRMATION: _CONFIRMATION_STYLE,
        SpeechPriority.DEBUG: _STATUS_STYLE,
    }
    
    # Error state isn't a priority per se, but fallback to STATUS if unknown
    return mapping.get(priority, _STATUS_STYLE)
