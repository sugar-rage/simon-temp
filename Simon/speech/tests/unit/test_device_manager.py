"""
Unit tests for the audio DeviceManager.

Tests device enumeration, ranking logic, loopback filtering,
and the device data model — all without requiring a real microphone.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speech.audio.device_manager import (
    AudioDevice,
    DeviceManager,
    _LOOPBACK_KEYWORDS,
    _PREFERRED_BACKENDS,
)
from speech.errors.exceptions import DeviceNotFoundError


# ---------------------------------------------------------------------- #
#  AudioDevice model tests
# ---------------------------------------------------------------------- #

class TestAudioDevice:
    """Tests for the AudioDevice dataclass."""

    def test_backend_priority_mme(self):
        dev = AudioDevice(device_id=0, name="Mic", host_api="MME")
        assert dev.backend_priority == 0  # Most preferred on Windows

    def test_backend_priority_wasapi(self):
        dev = AudioDevice(device_id=0, name="Mic", host_api="Windows WASAPI")
        assert dev.backend_priority == 1

    def test_backend_priority_directsound(self):
        dev = AudioDevice(device_id=0, name="Mic", host_api="Windows DirectSound")
        assert dev.backend_priority == 2

    def test_backend_priority_wdm_ks(self):
        dev = AudioDevice(device_id=0, name="Mic", host_api="Windows WDM-KS")
        assert dev.backend_priority == 3

    def test_backend_priority_unknown(self):
        dev = AudioDevice(device_id=0, name="Mic", host_api="ALSA")
        assert dev.backend_priority == len(_PREFERRED_BACKENDS)

    def test_default_values(self):
        dev = AudioDevice(device_id=5, name="Test Mic")
        assert dev.device_id == 5
        assert dev.is_real_mic is True
        assert dev.is_working is False
        assert dev.working_rate == 16000


# ---------------------------------------------------------------------- #
#  Loopback filtering tests
# ---------------------------------------------------------------------- #

class TestLoopbackFiltering:
    """Tests that loopback/virtual devices are correctly identified."""

    @pytest.mark.parametrize("name", [
        "Stereo Mix",
        "Speakers (loopback)",
        "PC Speaker",
        "What U Hear",
        "Headphone Output",
    ])
    def test_loopback_device_detected(self, name):
        is_loopback = any(kw in name.lower() for kw in _LOOPBACK_KEYWORDS)
        assert is_loopback is True

    @pytest.mark.parametrize("name", [
        "Microphone (Realtek)",
        "USB Audio Device",
        "Jabra SPEAK 510",
        "Blue Yeti",
    ])
    def test_real_mic_not_filtered(self, name):
        is_loopback = any(kw in name.lower() for kw in _LOOPBACK_KEYWORDS)
        assert is_loopback is False


# ---------------------------------------------------------------------- #
#  DeviceManager tests
# ---------------------------------------------------------------------- #

class TestDeviceManager:
    """Tests for DeviceManager logic (mocked sounddevice)."""

    def test_init_default_params(self):
        dm = DeviceManager()
        assert dm.current_device is None

    def test_init_custom_params(self):
        dm = DeviceManager(
            preferred_device_id=3,
            target_sample_rate=44100,
            poll_interval_s=5.0,
        )
        assert dm._preferred_id == 3
        assert dm._target_rate == 44100

    def test_enumerate_filters_output_devices(self):
        """Devices with max_input_channels=0 should be excluded."""
        sd = pytest.importorskip("sounddevice", reason="sounddevice not installed")

        dm = DeviceManager()

        # Mock sounddevice — provide a mix of input and output devices
        mock_devices = [
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100},
        ]
        mock_hostapis = [{"name": "MME", "devices": [0, 1]}]

        with patch.object(sd, "query_devices", return_value=mock_devices), \
             patch.object(sd, "query_hostapis", return_value=mock_hostapis):
            result = dm.enumerate_devices()

        # Only the microphone should be returned
        assert len(result) == 1
        assert result[0].name == "Microphone"

    def test_get_best_device_raises_when_none_found(self):
        """Should raise DeviceNotFoundError when no devices work."""
        dm = DeviceManager()

        with patch.object(dm, "enumerate_devices", return_value=[]), \
             patch.object(dm, "probe_device", return_value=False):
            with pytest.raises(DeviceNotFoundError):
                dm.get_best_device()

    def test_device_ranking_prefers_real_mics(self):
        """Real mics should be ranked before loopback devices."""
        devices = [
            AudioDevice(device_id=0, name="Stereo Mix", host_api="MME", is_real_mic=False),
            AudioDevice(device_id=1, name="USB Mic", host_api="MME", is_real_mic=True),
        ]
        # Sort as DeviceManager does
        devices.sort(key=lambda d: (not d.is_real_mic, d.backend_priority))
        assert devices[0].name == "USB Mic"

    def test_device_ranking_prefers_better_backend(self):
        """MME should be preferred over WDM-KS for the same mic on Windows."""
        devices = [
            AudioDevice(device_id=0, name="Mic (MME)", host_api="MME", is_real_mic=True),
            AudioDevice(device_id=1, name="Mic (WDM-KS)", host_api="Windows WDM-KS", is_real_mic=True),
        ]
        devices.sort(key=lambda d: (not d.is_real_mic, d.backend_priority))
        assert devices[0].host_api == "MME"

    def test_event_callback_registration(self):
        dm = DeviceManager()
        callback = MagicMock()
        dm.on_device_connected(callback)
        dm.on_device_disconnected(callback)
        dm.on_device_error(callback)
        assert len(dm._on_device_connected) == 1
        assert len(dm._on_device_disconnected) == 1
        assert len(dm._on_device_error) == 1
