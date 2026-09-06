"""
Speak Pipeline.

Manages the background thread that consumes tasks from the SpeechPriorityQueue
and plays audio via sounddevice. Supports preemption (interrupting playback)
if a higher priority message arrives.
"""

from __future__ import annotations

import logging
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from speech.monitoring.logger import get_logger
from speech.queue.priority_queue import SpeechPriorityQueue, SpeakItem
from speech.tts.streaming_synthesizer import StreamingSynthesizer
from speech.tts.emotional_profile import get_style_for_priority

if TYPE_CHECKING:
    from speech.manager.speech_manager import SpeechManager

logger = get_logger("manager.speak_pipeline")


class SpeakPipeline:
    """Consumes the priority queue and plays synthesized audio."""

    def __init__(
        self,
        manager: 'SpeechManager',
        synthesizer: StreamingSynthesizer,
        priority_queue: SpeechPriorityQueue,
        output_device_id: Optional[int] = None
    ):
        self._manager = manager
        self._synthesizer = synthesizer
        self._queue = priority_queue
        self._output_device_id = output_device_id
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

    def start(self) -> None:
        """Start the speak pipeline background thread."""
        if self._is_running:
            return
            
        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="SpeakPipelineThread",
            daemon=True
        )
        self._thread.start()
        logger.info("Speak pipeline started.")

    def stop(self) -> None:
        """Stop the speak pipeline background thread."""
        if not self._is_running:
            return
            
        self._is_running = False
        self._stop_event.set()
        self._queue.abort_all()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
        logger.info("Speak pipeline stopped.")

    def _run_loop(self) -> None:
        """Main loop consuming the priority queue."""
        import sounddevice as sd
        
        while not self._stop_event.is_set():
            try:
                # Block until an item is available
                # Using a timeout allows checking _stop_event periodically
                item: SpeakItem = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if self._stop_event.is_set() or item.aborted:
                self._queue.task_done()
                continue
                
            logger.info(f"[SPEECH] TTS item dequeued: {item.text!r} (priority={item.priority})")
            logger.info(f"[SPEECH] TTS started: {item.text}")
            
            # Pause microphone input while speaking to prevent self-triggering
            if self._manager._listen_pipeline:
                try:
                    self._manager._listen_pipeline.pause()
                    logger.info("[SPEECH] STT paused for TTS")
                except Exception as e:
                    logger.warning(f"[SPEECH] Failed to pause STT: {e}")
                
            try:
                style = get_style_for_priority(item.priority)
                logger.info(f"[SPEECH] Synthesis started: {item.text!r}")
                logger.info(f"[TTS_TRACE] text requested: {item.text!r}")
                logger.info(f"[TTS_TRACE] synthesis started: {item.text!r}")
                
                chunks = []
                for chunk_idx, chunk in enumerate(self._synthesizer.synthesize_stream(item.text, style)):
                    if self._stop_event.is_set() or self._queue.abort_event.is_set():
                        logger.info("[SPEECH] Playback preempted during synthesis.")
                        logger.info(f"[TTS_TRACE] playback preempted before chunk {chunk_idx + 1}")
                        break
                        
                    if chunk.data is not None and len(chunk.data) > 0:
                        chunks.append(chunk)
                        chunk_dur = len(chunk.data) / float(chunk.sample_rate)
                        logger.info(f"[SPEECH] Synthesized chunk {chunk_idx + 1}")
                        logger.info(
                            f"[TTS_TRACE] synthesis chunk #{chunk_idx + 1}: samples={len(chunk.data)} duration={chunk_dur:.2f}s"
                        )

                total_chunks = len(chunks)
                logger.info(f"[TTS_TRACE] total chunks={total_chunks}")

                if total_chunks > 0 and not (self._stop_event.is_set() or self._queue.abort_event.is_set()):
                    sample_rate = chunks[0].sample_rate
                    # Concatenate all PCM chunks into one contiguous buffer for seamless playback
                    combined_audio = np.concatenate([c.data for c in chunks])
                    total_samples = len(combined_audio)
                    total_duration = total_samples / float(sample_rate)

                    logger.info(f"[TTS_TRACE] total synthesized samples={total_samples}")
                    logger.info(f"[TTS_TRACE] total synthesized duration={total_duration:.2f}s")

                    try:
                        out_dev_id = self._output_device_id
                        dev_info = {}
                        hostapi_name = "Default"
                        try:
                            if out_dev_id is None:
                                default_devices = sd.default.device
                                if isinstance(default_devices, (list, tuple)) and len(default_devices) > 1:
                                    out_dev_id = default_devices[1]
                            if out_dev_id is not None:
                                dev_info = sd.query_devices(out_dev_id)
                                hostapi_name = sd.query_hostapis(dev_info.get("hostapi", 0)).get("name", "Unknown")
                        except Exception as e:
                            logger.warning(f"[TTS_TRACE] Failed to query output device info: {e}")

                        dev_name = dev_info.get("name", "Default Output")
                        logger.info(
                            f"[TTS_TRACE] output device selected: index={out_dev_id}, name={dev_name!r}, host_api={hostapi_name!r}, sample_rate={sample_rate}"
                        )
                        logger.info(f"[TTS_TRACE] samples submitted to PortAudio: {total_samples} samples ({total_duration:.2f}s)")
                        logger.info(f"[SPEECH] Playback started: total {total_samples} samples ({total_duration:.2f}s)")
                        logger.info(f"[TTS_TRACE] playback chunk #1 started")
                        logger.info(f"[TTS_TRACE] playback started (samples={total_samples}, duration={total_duration:.2f}s)")

                        t_stream_start = time.monotonic()
                        # Use dedicated OutputStream for guaranteed uninterrupted output
                        try:
                            with sd.OutputStream(
                                samplerate=sample_rate,
                                channels=1,
                                dtype="int16",
                                device=self._output_device_id,
                            ) as stream:
                                logger.info(f"[TTS_TRACE] PortAudio stream start (active={stream.active})")
                                stream.write(combined_audio)
                                t_write_done = time.monotonic()
                                write_elapsed = t_write_done - t_stream_start
                                logger.info(f"[TTS_TRACE] PortAudio stream write complete (active={stream.active}, write_time={write_elapsed:.2f}s)")
                                
                                # Hardware clock drain: ensure PortAudio hardware DAC physically outputs all samples
                                remaining_playback = total_duration - write_elapsed
                                if remaining_playback > 0:
                                    logger.info(f"[TTS_TRACE] hardware DAC drain waiting: {remaining_playback:.2f}s")
                                    time.sleep(remaining_playback + 0.05)
                        except Exception as stream_err:
                            # Fallback to sd.play if OutputStream directly fails or in mocked test environment
                            logger.warning(f"[TTS_TRACE] OutputStream write failed ({stream_err}), falling back to sd.play: {stream_err}")
                            sd.play(
                                combined_audio,
                                samplerate=sample_rate,
                                blocking=True,
                                device=self._output_device_id,
                            )
                            sd.wait()

                        t_stream_end = time.monotonic()
                        total_playback_time = t_stream_end - t_stream_start
                        logger.info(f"[TTS_TRACE] PortAudio stream closed (total_time={total_playback_time:.2f}s)")
                        logger.info(f"[SPEECH] Playback completed: total {total_samples} samples ({total_duration:.2f}s)")
                        logger.info(f"[TTS_TRACE] playback chunk #1 completed")
                        logger.info(f"[TTS_TRACE] playback total completed")
                    except Exception as e:
                        logger.error(f"[SPEECH] TTS playback error: {e}", exc_info=True)
                        logger.error(f"[TTS_TRACE] playback error: {e}")

                # Ensure all audio buffers have finished physical playback
                try:
                    sd.wait()
                except Exception:
                    pass

                # Post-playback acoustic drain cooldown: guarantees room echo / physical speaker output has completely ended before STT unpauses
                logger.info(f"[TTS_TRACE] drain started (duration=0.25s)")
                time.sleep(0.25)
                logger.info(f"[TTS_TRACE] drain completed")
                
                logger.info(f"[SPEECH] Synthesis completed: {item.text!r}")
                logger.info(f"[SPEECH] TTS completed: {item.text}")
                logger.info(f"[TTS_TRACE] TTS completed: {item.text!r}")
                if item.on_complete:
                    try:
                        item.on_complete(True)
                    except Exception as exc:
                        logger.warning(f"[TTS_TRACE] on_complete callback error: {exc}")
                            
            except Exception as e:
                logger.error(f"[SPEECH] TTS error: {e}", exc_info=True)
                logger.error(f"[TTS_TRACE] TTS exception: {e}")
                if item.on_complete:
                    try:
                        item.on_complete(False)
                    except Exception as exc:
                        logger.warning(f"[TTS_TRACE] on_complete error callback error: {exc}")
                
            finally:
                self._queue.task_done()
                
                # Resume listening only after physical playback and acoustic drain have completed
                if self._manager._listen_pipeline:
                    try:
                        self._manager._listen_pipeline.resume()
                        logger.info("[SPEECH] STT resumed after TTS")
                        logger.info("[TTS_TRACE] STT resumed")
                    except Exception as e:
                        logger.warning(f"[SPEECH] Failed to resume STT: {e}")
                        logger.warning(f"[TTS_TRACE] Failed to resume STT: {e}")
