"""
Noise suppressor — DeepFilterNet 3 integration with spectral subtraction fallback.

DeepFilterNet 3 is a state-of-the-art deep-learning noise suppression model
that operates in real-time.  It dramatically improves ASR accuracy in noisy
environments by removing background noise while preserving speech.

Design decision: DeepFilterNet operates at 48kHz internally.  We resample
in/out transparently so the rest of the pipeline can work at 16kHz.  If
DeepFilterNet is unavailable (not installed, GPU OOM), we fall back to a
simple spectral subtraction algorithm — inferior quality but better than
no suppression.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from speech.models.audio_frame import AudioFrame
from speech.monitoring.logger import get_logger
from speech.monitoring.metrics import get_collector

logger = get_logger("audio.noise_suppressor")
metrics = get_collector()


class NoiseSuppressor:
    """Removes background noise from audio frames.

    Attempts to load DeepFilterNet 3 on init.  If unavailable, falls back
    to spectral subtraction.

    Args:
        attenuation_db: Target noise attenuation in dB (for spectral fallback).
        enable_deep_filter: Whether to attempt loading DeepFilterNet.
    """

    def __init__(
        self,
        attenuation_db: float = 20.0,
        enable_deep_filter: bool = True,
    ):
        self._attenuation_db = attenuation_db
        self._df_model = None
        self._df_state = None
        self._active_backend = "none"

        if enable_deep_filter:
            self._try_load_deepfilter()

        if self._df_model is None:
            logger.info("Using spectral subtraction fallback for noise suppression")
            self._active_backend = "spectral_subtraction"
            # Running noise estimate for spectral subtraction
            self._noise_estimate: Optional[np.ndarray] = None
            self._noise_alpha = 0.98  # Smoothing factor

    def _try_load_deepfilter(self) -> None:
        """Attempt to load DeepFilterNet 3."""
        try:
            from df.enhance import enhance, init_df

            self._df_model, self._df_state, _ = init_df()
            self._active_backend = "deepfilternet3"
            logger.info("DeepFilterNet 3 loaded successfully")
        except ImportError:
            logger.info("DeepFilterNet not installed (pip install deepfilternet)")
        except Exception as e:
            logger.warning(f"DeepFilterNet init failed: {e}")

    @property
    def backend(self) -> str:
        """Name of the active noise suppression backend."""
        return self._active_backend

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Apply noise suppression to an audio frame.

        Args:
            frame: Input AudioFrame.

        Returns:
            Noise-suppressed AudioFrame.
        """
        if self._active_backend == "deepfilternet3":
            return self._process_deepfilter(frame)
        elif self._active_backend == "spectral_subtraction":
            return self._process_spectral(frame)
        else:
            return frame  # No suppression

    def _process_deepfilter(self, frame: AudioFrame) -> AudioFrame:
        """Process with DeepFilterNet 3."""
        try:
            from df.enhance import enhance

            # DeepFilterNet expects float32 at 48kHz
            f32 = frame.to_float32()
            resampled = f32.resample(48000)

            import torch
            audio_tensor = torch.from_numpy(resampled.data).unsqueeze(0)
            enhanced = enhance(self._df_model, self._df_state, audio_tensor)

            if isinstance(enhanced, torch.Tensor):
                enhanced_np = enhanced.squeeze().numpy()
            else:
                enhanced_np = np.array(enhanced).flatten()

            # Resample back to original rate
            result = AudioFrame(
                data=enhanced_np.astype(np.float32),
                sample_rate=48000,
                channels=1,
                dtype="float32",
                timestamp=frame.timestamp,
                device_id=frame.device_id,
                sequence_id=frame.sequence_id,
            )
            result = result.resample(frame.sample_rate)

            # Convert back to original dtype
            if frame.dtype == "int16":
                result = result.to_int16()

            return result

        except Exception as e:
            logger.warning(f"DeepFilterNet processing failed: {e}, falling back")
            metrics.counter("audio.deepfilter_errors")
            return self._process_spectral(frame)

    def _process_spectral(self, frame: AudioFrame) -> AudioFrame:
        """Simple spectral subtraction noise suppression.

        Estimates noise spectrum from quiet frames and subtracts it from
        speech frames.  Much simpler than DeepFilterNet but provides
        basic noise reduction.
        """
        f32 = frame.to_float32()
        audio = f32.data.astype(np.float64)

        # FFT
        n_fft = min(512, len(audio))
        if len(audio) < n_fft:
            return frame  # Too short for FFT

        spectrum = np.fft.rfft(audio[:n_fft])
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Update noise estimate (running average of magnitude)
        if self._noise_estimate is None or len(self._noise_estimate) != len(magnitude):
            self._noise_estimate = magnitude.copy()
        else:
            # Only update noise estimate when signal is relatively quiet
            rms = frame.rms_energy
            if rms < 200:  # Likely noise-only
                self._noise_estimate = (
                    self._noise_alpha * self._noise_estimate
                    + (1 - self._noise_alpha) * magnitude
                )

        # Spectral subtraction with flooring
        attenuation_factor = 10 ** (self._attenuation_db / 20)
        cleaned_magnitude = np.maximum(
            magnitude - self._noise_estimate * attenuation_factor,
            magnitude * 0.1,  # Floor at -20dB of original
        )

        # Reconstruct
        cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = np.fft.irfft(cleaned_spectrum, n=n_fft)

        # Pad or truncate to original length
        result = np.zeros_like(audio)
        result[:min(len(cleaned_audio), len(result))] = cleaned_audio[:len(result)]

        result_data = result.astype(np.float32)
        if frame.dtype == "int16":
            result_data = np.clip(result_data, -1.0, 1.0)
            result_data = (result_data * 32768.0).astype(np.int16)

        return AudioFrame(
            data=result_data,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            dtype=frame.dtype,
            timestamp=frame.timestamp,
            device_id=frame.device_id,
            sequence_id=frame.sequence_id,
        )
