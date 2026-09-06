"""
Continuous audio input stream with ring buffer and overflow protection.

Wraps ``sounddevice.InputStream`` to provide:
- Thread-safe, continuous audio capture
- Ring buffer to absorb latency spikes without dropping frames
- Automatic reconnection when the device disconnects
- Pause/resume for TTS feedback suppression
- Overflow detection and metrics reporting

Design decision: We use a callback-based InputStream (rather than blocking
``sd.rec()``) because it decouples audio capture from processing.  The
callback pushes frames into a thread-safe queue that the consumer reads
at its own pace.  This prevents the consumer's processing latency from
causing audio gaps.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np

from speech.audio.device_manager import AudioDevice
from speech.errors.exceptions import AudioOverflowError, DeviceDisconnectedError
from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector

logger = get_logger("audio.stream")
metrics = get_collector()


class AudioStream:
    """Continuous audio input stream with ring buffer.

    Args:
        device:       The audio device to capture from.
        sample_rate:  Sample rate in Hz (default: device's working rate).
        channels:     Number of channels (default: 1 = mono).
        frame_duration_ms: Duration of each frame in milliseconds.
        buffer_size:  Max frames in the ring buffer before overflow.
    """

    def __init__(
        self,
        device: AudioDevice,
        sample_rate: int | None = None,
        channels: int = 1,
        frame_duration_ms: int = 30,
        buffer_size: int = 500,
    ):
        self._device = device
        self._sample_rate = sample_rate or device.working_rate
        self._channels = channels
        self._frame_duration_ms = frame_duration_ms
        self._buffer_size = buffer_size

        # Compute block size from frame duration
        self._block_size = int(self._sample_rate * frame_duration_ms / 1000)

        # Thread-safe frame queue
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=buffer_size)

        # State
        self._stream = None
        self._running = False
        self._paused = False
        self._pause_lock = threading.Lock()
        self._frame_counter = 0
        self._overflow_count = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def device(self) -> AudioDevice:
        return self._device

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start capturing audio from the device."""
        if self._running:
            return

        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                device=self._device.device_id,
                channels=self._channels,
                dtype="int16",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            logger.info(
                f"Audio stream started",
                extra={
                    "device": self._device.name,
                    "rate": self._sample_rate,
                    "block_size": self._block_size,
                },
            )
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise DeviceDisconnectedError(self._device.device_id) from e

    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            self._stream = None

        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Audio stream stopped")

    def pause(self) -> None:
        """Pause audio capture (for TTS feedback suppression).

        Audio continues to flow through the hardware callback but
        frames are discarded instead of queued.
        """
        with self._pause_lock:
            self._paused = True
            # Drain queued frames so stale audio isn't processed on resume
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def resume(self) -> None:
        """Resume audio capture after pause."""
        with self._pause_lock:
            self._paused = False

    # ------------------------------------------------------------------ #
    #  Audio callback (runs on PortAudio's real-time thread)
    # ------------------------------------------------------------------ #

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback — pushes audio into the queue.

        This runs on a real-time thread and must be as fast as possible.
        No blocking operations, no logging, no allocations beyond the
        AudioFrame dataclass.
        """
        if status:
            # Report overflow/underflow via metrics (non-blocking)
            metrics.counter("audio.callback_status_errors")

        # If paused (TTS speaking), discard frame
        if self._paused:
            return

        self._frame_counter += 1

        frame = AudioFrame(
            data=indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy(),
            sample_rate=self._sample_rate,
            channels=1,
            dtype="int16",
            timestamp=time.monotonic(),
            device_id=self._device.device_id,
            sequence_id=self._frame_counter,
        )

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Overflow: drop oldest frame
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            self._overflow_count += 1
            metrics.counter("audio.dropped_frames")

    # ------------------------------------------------------------------ #
    #  Consumer API
    # ------------------------------------------------------------------ #

    def read_frame(self, timeout: float = 0.1) -> Optional[AudioFrame]:
        """Read the next audio frame from the buffer.

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            An ``AudioFrame``, or ``None`` if timeout expired.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_frames(self, timeout: float = 0.1) -> Iterator[AudioFrame]:
        """Yield audio frames continuously.

        Blocks up to ``timeout`` seconds for each frame.  Yields None
        on timeout (allowing the caller to check ``is_running``).
        """
        while self._running:
            frame = self.read_frame(timeout=timeout)
            if frame is not None:
                yield frame

    def drain(self) -> list[AudioFrame]:
        """Read all currently buffered frames without blocking."""
        frames = []
        while not self._queue.empty():
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return frames

    @property
    def buffered_frames(self) -> int:
        """Number of frames currently in the buffer."""
        return self._queue.qsize()

    @property
    def buffered_duration_s(self) -> float:
        """Estimated duration of buffered audio in seconds."""
        return self.buffered_frames * self._frame_duration_ms / 1000.0
