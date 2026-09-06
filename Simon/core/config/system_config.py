"""SystemConfig — typed configuration dataclass.

Provides typed, validated access to SIMON configuration values loaded
from YAML files.  Default values ensure the system can start without
any configuration file present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    """Camera subsystem configuration."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_reconnect: bool = True
    reconnect_delay_s: float = 2.0
    frame_buffer_size: int = 5


@dataclass
class DetectionConfig:
    """Object detection configuration."""

    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    device: str = "auto"  # "cuda", "cpu", or "auto"
    max_detections: int = 50


@dataclass
class OCRConfig:
    """OCR configuration."""

    primary_engine: str = "pytesseract"
    fallback_engine: str = "easyocr"
    languages: list[str] = field(default_factory=lambda: ["eng"])
    confidence_threshold: float = 0.3


@dataclass
class FaceConfig:
    """Face recognition configuration."""

    model: str = "buffalo_l"
    database_dir: str = "data/faces"
    similarity_threshold: float = 0.4
    detection_interval_frames: int = 5
    known_person_reannounce_timeout_s: float = 300.0


@dataclass
class DepthConfig:
    """Depth estimation configuration."""

    enabled: bool = False
    model: str = "MiDaS_small"
    device: str = "auto"


@dataclass
class VisionConfig:
    """Complete vision subsystem configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    scene_analysis_interval_frames: int = 30
    ocr_interval_frames: int = 10


@dataclass
class GPSConfig:
    """GPS configuration."""

    provider: str = "serial"  # "serial" or "simulation"
    port: str = "/dev/ttyUSB0"
    baud_rate: int = 9600
    simulation_route_file: Optional[str] = None


@dataclass
class RoutingConfig:
    """Routing configuration."""

    primary_router: str = "osrm"
    osrm_url: str = "http://router.project-osrm.org"
    fallback_router: str = "osm_graph"
    cache_dir: str = "data/routing_cache"


@dataclass
class GuidanceConfig:
    """Voice guidance configuration."""

    announce_distance_m: float = 30.0
    off_route_threshold_m: float = 25.0
    arrival_threshold_m: float = 10.0


@dataclass
class NavigationConfig:
    """Complete navigation subsystem configuration."""

    gps: GPSConfig = field(default_factory=GPSConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)


@dataclass
class PluginConfig:
    """Plugin system configuration."""

    enabled: bool = True
    plugin_dir: str = "plugins"
    auto_discover: bool = True


@dataclass
class SafetyConfig:
    """Safety engine configuration."""

    enabled: bool = True
    hazard_cooldown_s: float = 3.0
    critical_distance_m: float = 2.0
    warning_distance_m: float = 5.0
    vehicle_classes: list[str] = field(
        default_factory=lambda: [
            "car", "truck", "bus", "motorcycle", "bicycle",
        ]
    )


@dataclass
class SystemConfig:
    """Top-level SIMON system configuration.

    This is the root configuration object, assembled from
    multiple YAML files via deep merge.
    """

    # Subsystem configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Global settings
    log_level: str = "INFO"
    log_file: Optional[str] = "simon.log"
    log_dir: str = "logs"
    data_dir: str = "data"
    debug: bool = False

    # Thread settings
    event_bus_rate_limit_s: float = 0.05
    event_bus_queue_size: int = 10_000
    watchdog_interval_s: float = 5.0
    health_check_interval_s: float = 30.0
