"""
SIMON Configuration — Central settings for all modules.
Uses OpenStreetMap (OSRM + Nominatim) — completely FREE, no API key needed.
"""

# =====================================================================
#  ROUTING API (OpenStreetMap — FREE, no key required)
# =====================================================================
ROUTING_ENGINE = "osrm"                     # "osrm" (free) | "google" (needs key)
OSRM_SERVER = "https://router.project-osrm.org"  # Free public OSRM server
NOMINATIM_URL = "https://nominatim.openstreetmap.org"  # Free geocoding
NOMINATIM_USER_AGENT = "SIMON-Navigation/1.0"  # Required by Nominatim policy

# Google Maps (only if ROUTING_ENGINE = "google")
GOOGLE_MAPS_API_KEY = ""                    # Optional — leave empty to use OSRM

# =====================================================================
#  GPS SETTINGS
# =====================================================================
GPS_MODE = "serial"          # "serial" (USB dongle) | "gpsd" | "simulation"
GPS_SERIAL_PORT = "COM3"     # Windows COM port for USB GPS
GPS_BAUD_RATE = 9600
GPS_UPDATE_INTERVAL = 1.0    # Seconds between GPS reads

# Simulation mode — used when no real GPS is available
GPS_SIM_START_LAT = 12.9716
GPS_SIM_START_LON = 77.5946
GPS_SIM_SPEED = 1.2          # Simulated walking speed (m/s)

# =====================================================================
#  NAVIGATION
# =====================================================================
NAV_ARRIVAL_RADIUS = 10.0         # Meters — "you have arrived" radius
NAV_OFF_ROUTE_THRESHOLD = 30.0    # Meters — trigger re-route
NAV_TURN_WARN_DISTANCE = 15.0     # Meters — "turn left in X meters"
NAV_INSTRUCTION_COOLDOWN = 5.0    # Seconds — min gap between same instruction
NAV_RECALC_COOLDOWN = 15.0        # Seconds — min gap between re-routes
NAV_TRAVEL_MODE = "walking"       # "walking" | "driving" | "bicycling"

# =====================================================================
#  COMPASS
# =====================================================================
COMPASS_DEVIATION_THRESHOLD = 45  # Degrees off-course before warning
COMPASS_DIRECTION_NAMES = [
    "North", "North-East", "East", "South-East",
    "South", "South-West", "West", "North-West"
]

# =====================================================================
#  OBSTACLE DETECTION  —  OBJECT DETECTION DISABLED (temporary)
# =====================================================================
OBSTACLE_ENABLED = True   # Object detection enabled
# OBSTACLE_MODEL = "yolov8n.pt"     # YOLOv8 nano for speed
# OBSTACLE_CONFIDENCE = 0.45        # Min detection confidence
# OBSTACLE_NEAR_THRESHOLD = 0.35    # BBox height ratio → "NEAR" (< ~2m)
# OBSTACLE_MEDIUM_THRESHOLD = 0.18  # BBox height ratio → "MEDIUM" (2–5m)
# OBSTACLE_WARNING_COOLDOWN = 3.0   # Seconds between same-object warnings
#
# # Classes to detect as obstacles (COCO dataset IDs)
# OBSTACLE_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "bus", "truck",
#     "dog", "cat", "chair", "bench", "fire hydrant",
#     "stop sign", "parking meter", "potted plant",
#     "dining table", "suitcase", "backpack",
# ]

# =====================================================================
#  VOICE ENGINE (TTS) — Coqui Neural TTS + pyttsx3 fallback
# =====================================================================
TTS_ENGINE = "coqui"           # "coqui" | "pyttsx3" | "auto" (try coqui, fallback pyttsx3)
TTS_COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"   # Same model as working simon project
TTS_COQUI_SPEED = 0.85       # Speech speed (0.5 = slow, 1.0 = normal)
VOICE_RATE = 175              # pyttsx3 words per minute (fallback only)
VOICE_VOLUME = 1.0            # 0.0 – 1.0
VOICE_LANGUAGE = "en"

# Priority levels (lower number = higher priority)
VOICE_PRIORITY_OBSTACLE = 1
VOICE_PRIORITY_NAVIGATION = 2
VOICE_PRIORITY_STATUS = 3
VOICE_PRIORITY_FACE = 4

# =====================================================================
#  OCR — Tesseract / EasyOCR Scene Text Recognition
# =====================================================================
OCR_ENABLED = True
OCR_BACKEND = "auto"          # "pytesseract" | "easyocr" | "auto"
OCR_INTERVAL = 10             # Process OCR every N frames
OCR_CONFIDENCE = 0.40         # Minimum confidence to accept text
OCR_COOLDOWN = 12.0           # Seconds — avoid re-announcing same text
OCR_DESKEW = True             # Correct tilted text before OCR
OCR_LANGUAGE = "en"           # Language for easyocr
OCR_GPU = False               # Use GPU for easyocr (if available)
# Windows Tesseract path — update if installed in a different location
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =====================================================================
#  FACE RECOGNITION (InsightFace)
# =====================================================================
FACE_RECOGNITION_ENABLED = True
FACE_RECOGNITION_INTERVAL = 5 # Process faces every N frames
FACE_MATCH_THRESHOLD = 0.58   # Cosine similarity threshold for identity match
FACE_DATABASE_PATH = "SAVED_FACES" # Directory where known faces are stored

# =====================================================================
#  SPEECH INPUT — Whisper Voice Commands
# =====================================================================
WHISPER_ENABLED = True
MIC_DEVICE_INDEX = 1          # Set to override microphone auto-selection (e.g. 1)
WHISPER_MODEL = "base"        # "tiny" | "base" | "small" (larger = more accurate)
WHISPER_LANGUAGE = "en"
WHISPER_SAMPLE_RATE = 16000   # Hz — Whisper expects 16kHz
WHISPER_SILENCE_TIMEOUT = 1.5 # Seconds of silence before stopping recording
WHISPER_MAX_DURATION = 10     # Max recording duration (seconds)
WHISPER_VAD_AGGRESSIVENESS = 2  # 0 (least) to 3 (most aggressive)

# Barge-in control
# False (default): pause microphone listening while TTS is speaking to prevent
#                  the system from transcribing its own voice as a command.
# True:            keep listening while speaking (barge-in mode — user can
#                  interrupt Simon mid-sentence with a new command).
ENABLE_BARGE_IN = False

# =====================================================================
#  INDOOR NAVIGATION (Optional Advanced)
# =====================================================================
INDOOR_ENABLED = False
INDOOR_QR_ENABLED = True
INDOOR_BLE_ENABLED = False
INDOOR_BLE_SCAN_INTERVAL = 2.0    # Seconds
# Map of BLE beacon UUIDs → location names (fill in with your beacons)
INDOOR_BLE_BEACONS = {
    # "beacon-uuid-1": {"name": "Room 203", "floor": 2, "x": 10.0, "y": 5.0},
    # "beacon-uuid-2": {"name": "Elevator", "floor": 2, "x": 15.0, "y": 0.0},
}
# Map of QR code values → location info
INDOOR_QR_LOCATIONS = {
    # "QR_ROOM_203": {"name": "Room 203", "floor": 2, "direction": "left"},
    # "QR_ELEVATOR": {"name": "Elevator", "floor": 2, "direction": "ahead"},
}
