"""
Audio device manager — enumerates, ranks, monitors, and hot-swaps microphones.

Responsible for:
- Discovering all input audio devices via ``sounddevice``
- Ranking devices by backend preference (WDM-KS > WASAPI > DirectSound > MME)
- Filtering loopback/virtual devices (Stereo Mix, speakers, etc.)
- Probing device capability (sample rates, channels)
- Detecting hot-swap events (device plugged/unplugged)
- Health monitoring via periodic heartbeat

Design decision: Polling-based hot-swap detection rather than OS-level
notifications, because ``sounddevice``/PortAudio does not expose device
change callbacks on all platforms.  The poll interval is configurable
(default 3 seconds) — low enough for responsive recovery, high enough
to avoid CPU waste.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from speech.errors.exceptions import DeviceNotFoundError, DeviceDisconnectedError
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector

logger = get_logger("audio.device_manager")
metrics = get_collector()


# Keywords that identify loopback/virtual/output devices (not real microphones)
_LOOPBACK_KEYWORDS = frozenset([
    "speaker", "loopback", "stereo mix", "pc speaker",
    "output", "headphone", "what u hear", "wave out",
])

# Preferred backend order on Windows (most reliable first)
_PREFERRED_BACKENDS = [
    "MME",
    "Windows WASAPI",
    "Windows DirectSound",
    "Windows WDM-KS",
]


@dataclass
class AudioDevice:
    """Metadata for a discovered audio input device.

    Attributes:
        device_id:     PortAudio device index.
        name:          Human-readable device name.
        host_api:      Audio backend name (e.g. "Windows WASAPI").
        max_channels:  Maximum supported input channels.
        default_rate:  Default sample rate reported by the driver.
        is_real_mic:   True if this appears to be a real microphone.
        is_working:    True if a test recording succeeded.
        working_rate:  The sample rate that was successfully tested.
    """

    device_id: int
    name: str
    host_api: str = ""
    max_channels: int = 1
    default_rate: int = 44100
    is_real_mic: bool = True
    is_working: bool = False
    working_rate: int = 16000

    @property
    def backend_priority(self) -> int:
        """Lower number = more preferred backend."""
        try:
            return _PREFERRED_BACKENDS.index(self.host_api)
        except ValueError:
            return len(_PREFERRED_BACKENDS)


class DeviceManager:
    """Manages audio input device discovery, selection, and health monitoring.

    Provides:
    - ``get_best_device()`` — returns the best available microphone
    - ``start_monitoring()`` — starts background hot-swap detection
    - Event callbacks for device connect/disconnect

    Args:
        preferred_device_id: If set, try this device first (from config).
        target_sample_rate:  Preferred sample rate (default 16000 for Whisper).
        poll_interval_s:     How often to poll for device changes.
    """

    def __init__(
        self,
        preferred_device_id: Optional[int] = None,
        target_sample_rate: int = 16000,
        poll_interval_s: float = 3.0,
    ):
        self._preferred_id = preferred_device_id
        self._target_rate = target_sample_rate
        self._poll_interval = poll_interval_s

        self._current_device: Optional[AudioDevice] = None
        self._known_devices: dict[int, AudioDevice] = {}
        self._lock = threading.Lock()

        # Event callbacks
        self._on_device_connected: list[Callable[[AudioDevice], None]] = []
        self._on_device_disconnected: list[Callable[[AudioDevice], None]] = []
        self._on_device_error: list[Callable[[AudioDevice, Exception], None]] = []

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------ #
    #  Event registration
    # ------------------------------------------------------------------ #

    def on_device_connected(self, callback: Callable[[AudioDevice], None]) -> None:
        """Register a callback for when a new device is detected."""
        self._on_device_connected.append(callback)

    def on_device_disconnected(self, callback: Callable[[AudioDevice], None]) -> None:
        """Register a callback for when a device is lost."""
        self._on_device_disconnected.append(callback)

    def on_device_error(self, callback: Callable[[AudioDevice, Exception], None]) -> None:
        """Register a callback for device errors."""
        self._on_device_error.append(callback)

    # ------------------------------------------------------------------ #
    #  Device discovery
    # ------------------------------------------------------------------ #

    def enumerate_devices(self) -> list[AudioDevice]:
        """Discover all input audio devices and return ranked list.

        Returns:
            List of ``AudioDevice`` sorted by preference (best first).
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed")
            return []

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        # Build host_api name lookup
        api_names = {}
        for idx, api in enumerate(host_apis):
            for dev_id in api["devices"]:
                api_names[dev_id] = api["name"]

        result: list[AudioDevice] = []
        for dev_id, dev in enumerate(devices):
            if dev["max_input_channels"] <= 0:
                continue

            name = dev["name"]
            host_api = api_names.get(dev_id, "")
            is_real = not any(kw in name.lower() for kw in _LOOPBACK_KEYWORDS)

            audio_dev = AudioDevice(
                device_id=dev_id,
                name=name,
                host_api=host_api,
                max_channels=dev["max_input_channels"],
                default_rate=int(dev["default_samplerate"]),
                is_real_mic=is_real,
            )
            result.append(audio_dev)

        # Sort: real mics first, then by backend priority
        result.sort(key=lambda d: (not d.is_real_mic, d.backend_priority))

        logger.info(
            f"Enumerated {len(result)} input devices",
            extra={"real_mics": sum(1 for d in result if d.is_real_mic)},
        )
        return result

    def probe_device(self, device: AudioDevice) -> bool:
        """Test if a device can actually record audio.

        Tries the target sample rate first (16kHz for Whisper), then
        falls back to the device's default rate.

        Returns:
            True if recording succeeded, updates device.is_working and
            device.working_rate.
        """
        try:
            import sounddevice as sd
        except ImportError:
            return False

        for test_rate in (self._target_rate, device.default_rate):
            try:
                test_audio = sd.rec(
                    int(0.1 * test_rate),
                    samplerate=test_rate,
                    channels=1,
                    dtype="int16",
                    device=device.device_id,
                )
                sd.wait()
                device.is_working = True
                device.working_rate = test_rate
                return True
            except Exception:
                continue

        device.is_working = False
        return False

    def get_best_device(self) -> AudioDevice:
        """Find and return the best available microphone.

        If a preferred device is configured and working, use it.
        Otherwise, enumerate and probe all devices, returning the first
        that works.

        Raises:
            DeviceNotFoundError: If no working microphone is found.
        """
        # Try preferred device first
        if self._preferred_id is not None:
            try:
                import sounddevice as sd
                dev_info = sd.query_devices(self._preferred_id)
                host_apis = sd.query_hostapis()
                api_name = ""
                for api in host_apis:
                    if self._preferred_id in api["devices"]:
                        api_name = api["name"]
                        break

                preferred = AudioDevice(
                    device_id=self._preferred_id,
                    name=dev_info["name"],
                    host_api=api_name,
                    max_channels=dev_info["max_input_channels"],
                    default_rate=int(dev_info["default_samplerate"]),
                )
                if self.probe_device(preferred):
                    logger.info(
                        f"Using configured device: [{preferred.device_id}] {preferred.name}",
                        extra={"rate": preferred.working_rate},
                    )
                    with self._lock:
                        self._current_device = preferred
                    metrics.label("audio.device", preferred.name)
                    return preferred
                else:
                    logger.warning(
                        f"Configured device {self._preferred_id} failed, falling back"
                    )
            except Exception as e:
                logger.warning(f"Configured device {self._preferred_id} error: {e}")

        # Enumerate and probe all devices
        all_devices = self.enumerate_devices()
        for device in all_devices:
            if self.probe_device(device):
                logger.info(
                    f"Auto-selected device: [{device.device_id}] {device.name}",
                    extra={
                        "rate": device.working_rate,
                        "backend": device.host_api,
                        "is_real_mic": device.is_real_mic,
                    },
                )
                with self._lock:
                    self._current_device = device
                    self._known_devices[device.device_id] = device
                metrics.label("audio.device", device.name)
                metrics.gauge("audio.sample_rate", device.working_rate)
                return device

        raise DeviceNotFoundError()

    @property
    def current_device(self) -> Optional[AudioDevice]:
        """The currently active audio device, or None."""
        with self._lock:
            return self._current_device

    # ------------------------------------------------------------------ #
    #  Health monitoring and hot-swap detection
    # ------------------------------------------------------------------ #

    def start_monitoring(self) -> None:
        """Start background thread for device health and hot-swap monitoring."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="device-monitor"
        )
        self._monitor_thread.start()
        logger.info("Device monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Device monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background loop: check device health, detect hot-swap."""
        while self._running:
            try:
                self._check_device_health()
                self._check_for_new_devices()
            except Exception as e:
                logger.error(f"Device monitor error: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(int(self._poll_interval * 10)):
                if not self._running:
                    return
                time.sleep(0.1)

    def _check_device_health(self) -> None:
        """Verify the current device is still connected and available without interrupting active audio."""
        with self._lock:
            current = self._current_device
        if current is None:
            return

        try:
            import sounddevice as sd
            devices = sd.query_devices()
            if current.device_id >= len(devices):
                device_still_present = False
            else:
                dev_info = devices[current.device_id]
                device_still_present = (
                    dev_info.get("max_input_channels", 0) > 0
                    and dev_info.get("name") == current.name
                )
        except Exception as e:
            logger.warning(f"Error querying audio devices during health check: {e}")
            device_still_present = False

        if not device_still_present:
            logger.warning(
                f"Device [{current.device_id}] {current.name} is no longer connected"
            )
            metrics.counter("audio.device_disconnects")

            for cb in self._on_device_disconnected:
                try:
                    cb(current)
                except Exception as e:
                    logger.error(f"Device disconnect callback error: {e}")

            # Attempt to find a replacement
            try:
                new_device = self.get_best_device()
                logger.info(f"Switched to device: [{new_device.device_id}] {new_device.name}")
                metrics.counter("audio.device_switches")
                for cb in self._on_device_connected:
                    try:
                        cb(new_device)
                    except Exception as e:
                        logger.error(f"Device connect callback error: {e}")
            except DeviceNotFoundError:
                logger.error("No replacement device found")
                with self._lock:
                    self._current_device = None

    def _check_for_new_devices(self) -> None:
        """Detect newly connected devices without interrupting active audio playback/capture."""
        current_devices = self.enumerate_devices()
        with self._lock:
            known_ids = set(self._known_devices.keys())

        for dev in current_devices:
            if dev.device_id not in known_ids:
                dev.is_working = True
                dev.working_rate = dev.default_rate
                with self._lock:
                    self._known_devices[dev.device_id] = dev
                logger.info(f"New device detected: [{dev.device_id}] {dev.name}")
                metrics.counter("audio.device_connects")
                for cb in self._on_device_connected:
                    try:
                        cb(dev)
                    except Exception as e:
                        logger.error(f"Device connect callback error: {e}")
