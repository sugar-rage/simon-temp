"""
SIMON Speech Input — Whisper-based voice command recognition.

Capabilities:
  - Background microphone listening with Voice Activity Detection (VAD)
  - Automatic speech start/stop detection (no push-to-talk needed)
  - Direct in-memory transcription using OpenAI Whisper (no disk I/O)
  - Flexible command parsing with optional wake word ("Simon")
  - Auto-detects working microphone across Windows audio backends
  - Thread-safe, runs alongside camera and navigation

Usage:
    speech = SpeechInput(voice_engine=engine)
    speech.start()
    # ... in main loop:
    command = speech.get_command()  # Non-blocking
    if command:
        handle_command(command)
    speech.stop()
"""

import threading
import queue
import time
import numpy as np

try:
    import config
except ImportError:
    config = None


# ======================================================================
#  Command Definitions
# ======================================================================
COMMAND_MAP = {
    # Navigation commands
    "navigate": "navigate",
    "navigate to": "navigate",
    "go to": "navigate",
    "take me to": "navigate",
    "directions to": "navigate",
    "start navigation": "navigate",

    # OCR commands
    "read": "read_text",
    "read text": "read_text",
    "what does it say": "read_text",
    "scan text": "read_text",
    "read sign": "read_text",
    "what is written": "read_text",

    # Face commands
    "save face": "save_face",
    "save this face": "save_face",
    "who is this": "save_face",
    "remember this face": "save_face",

    # Status
    "status": "status",
    "what's the status": "status",
    "what is the status": "status",
    "what is my status": "status",
    "where am i": "status",
    "current status": "status",

    # Indoor
    "indoor": "toggle_indoor",
    "indoor mode": "toggle_indoor",
    "switch to indoor": "toggle_indoor",

    # Control
    "stop": "stop",
    "quit": "stop",
    "exit": "stop",
    "cancel": "cancel_nav",
    "stop navigation": "cancel_nav",
}


class SpeechInput:
    """
    Whisper-based voice command input with VAD (Voice Activity Detection).

    Records when speech is detected, stops recording after silence,
    then transcribes directly in memory with Whisper (no disk I/O).
    """

    def __init__(self, voice_engine=None):
        """
        Initialize the speech input module.

        Args:
            voice_engine: VoiceEngine instance for audio feedback.
        """
        self.voice = voice_engine
        self._thread = None
        self._running = False
        self._enabled = False  # Toggle with V key
        self._command_queue = queue.Queue(maxsize=10)

        # Whisper model — configurable: "tiny" for low-RAM, "small" for accuracy
        self._model = None
        self._model_name = getattr(config, "WHISPER_MODEL", "small")
        self._language = getattr(config, "WHISPER_LANGUAGE", "en")

        # Audio settings
        self._whisper_rate = 16000  # Whisper always needs 16kHz
        self._sample_rate = 16000   # Updated to device native rate on init
        self._frame_duration_ms = 30
        self._silence_timeout = getattr(config, "WHISPER_SILENCE_TIMEOUT", 1.5)
        self._max_duration = getattr(config, "WHISPER_MAX_DURATION", 10)

        # Silence detection — RMS energy threshold (lower = more sensitive)
        self._silence_rms_threshold = getattr(config, "WHISPER_SILENCE_RMS", 30)

        # Audio device (auto-detected)
        self._audio_device = None

        # Wake word — set to "" to disable
        self._wake_word = getattr(config, "WAKE_WORD", "simon").lower().strip()

        # VAD
        self._vad = None
        self._vad_mode = getattr(config, "WHISPER_VAD_AGGRESSIVENESS", 2)

        # TTS feedback suppression
        # Set by pause_for_tts(), cleared by resume_after_tts().
        # When set (and ENABLE_BARGE_IN is False) the VAD loop discards audio
        # frames so Simon's own voice is never transcribed as a command.
        self._tts_speaking = threading.Event()
        self._barge_in = getattr(config, "ENABLE_BARGE_IN", False)

        # State
        self._backend_ready = False

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the background listening thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[SpeechInput] Started.")

    def stop(self):
        """Stop listening."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        print("[SpeechInput] Stopped.")

    def toggle(self):
        """Toggle voice command listening on/off."""
        self._enabled = not self._enabled
        state = "enabled" if self._enabled else "disabled"
        print(f"[SpeechInput] Voice commands: {state}")
        if self.voice:
            self.voice.speak(f"Voice commands {state}.", priority=3)
        return self._enabled

    # ------------------------------------------------------------------
    # TTS feedback suppression
    # ------------------------------------------------------------------
    def pause_for_tts(self):
        """
        Signal that TTS has started speaking.
        Called by VoiceEngine before runAndWait(). Thread-safe.
        """
        self._tts_speaking.set()

    def resume_after_tts(self):
        """
        Signal that TTS has finished speaking.
        Called by VoiceEngine in its finally block. Thread-safe.
        """
        self._tts_speaking.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_command(self):
        """
        Non-blocking: get the next parsed command, or None.

        Returns:
            dict: {"action": str, "args": str} or None
        """
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal — Initialization
    # ------------------------------------------------------------------
    def _find_working_device(self):
        """
        Try each input device until one successfully records audio.
        Prefers real microphones over loopback/speaker devices.
        Prefers WDM-KS > WASAPI > DirectSound > MME on Windows.
        Returns (device_id, native_sample_rate) or (None, None).
        """
        import sounddevice as sd

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        # Names that indicate loopback/internal devices (not real mics)
        loopback_keywords = ["speaker", "loopback", "stereo mix", "pc speaker",
                             "output", "headphone"]

        def is_real_mic(name):
            """Check if device name looks like a real microphone."""
            name_lower = name.lower()
            return not any(kw in name_lower for kw in loopback_keywords)

        # Build list of input devices, prioritising reliable backends
        preferred_order = ["Windows WDM-KS", "Windows WASAPI",
                           "Windows DirectSound", "MME"]
        real_mics = []
        loopback_mics = []

        for api_name in preferred_order:
            for idx, api in enumerate(host_apis):
                if api["name"] == api_name:
                    for dev_id in api["devices"]:
                        dev = devices[dev_id]
                        if dev["max_input_channels"] > 0:
                            entry = (dev_id, dev["name"], int(dev["default_samplerate"]))
                            if is_real_mic(dev["name"]):
                                real_mics.append(entry)
                            else:
                                loopback_mics.append(entry)

        # Also add any remaining input devices not yet covered
        covered_ids = {d[0] for d in real_mics + loopback_mics}
        for dev_id, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and dev_id not in covered_ids:
                entry = (dev_id, dev["name"], int(dev["default_samplerate"]))
                if is_real_mic(dev["name"]):
                    real_mics.append(entry)
                else:
                    loopback_mics.append(entry)

        # Try real microphones first, then fall back to loopback
        for dev_id, dev_name, native_rate in real_mics + loopback_mics:
            # Try 16000Hz natively first to avoid resampling
            for test_rate in (16000, native_rate):
                try:
                    test_audio = sd.rec(
                        int(0.1 * test_rate),
                        samplerate=test_rate,
                        channels=1,
                        dtype="int16",
                        device=dev_id,
                    )
                    sd.wait()
                    return dev_id, test_rate
                except Exception:
                    continue
        return None, None

    def _resample(self, audio, orig_rate, target_rate):
        """Resample audio from orig_rate to target_rate."""
        if orig_rate == target_rate:
            return audio
        from scipy.signal import resample
        num_samples = int(len(audio) * target_rate / orig_rate)
        return resample(audio, num_samples).astype(np.int16)

    def _init_backend(self):
        """Load Whisper model, find working mic, and init VAD."""
        try:
            import warnings
            warnings.filterwarnings("ignore")

            import whisper
            print(f"[SpeechInput] Loading Whisper model: '{self._model_name}' "
                  f"(options: tiny, base, small, medium)")
            self._model = whisper.load_model(self._model_name)
            print("[SpeechInput] Whisper model loaded OK")

        except Exception as e:
            print(f"[SpeechInput] Whisper unavailable: {e}")
            return False

        # Try to load VAD (webrtcvad) — optional
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._vad_mode)
            print(f"[SpeechInput] VAD ready (aggressiveness={self._vad_mode})")
        except ImportError:
            print("[SpeechInput] webrtcvad not installed — using timer-based recording")
            self._vad = None

        # Auto-detect a working microphone device
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError:
            print("[SpeechInput] sounddevice not installed!")
            return False

        mic_override = getattr(config, "MIC_DEVICE_INDEX", None)
        if mic_override is not None:
            try:
                dev_info = sd.query_devices(mic_override)
                # Prefer 16000Hz if supported to avoid resampling
                try:
                    sd.check_input_settings(device=mic_override, channels=1, dtype="int16", samplerate=16000)
                    native_rate = 16000
                except Exception:
                    native_rate = int(dev_info["default_samplerate"])
                
                self._audio_device = mic_override
                print(f"[SpeechInput] Using configured microphone: [{mic_override}] {dev_info['name']} at {native_rate}Hz")
            except Exception as e:
                print(f"[SpeechInput] Configured microphone {mic_override} failed: {e}. Falling back to auto-select.")
                self._audio_device, native_rate = self._find_working_device()
                if self._audio_device is not None:
                    dev_info = sd.query_devices(self._audio_device)
                    print(f"[SpeechInput] Auto-selected microphone: [{self._audio_device}] {dev_info['name']}")
        else:
            self._audio_device, native_rate = self._find_working_device()
            if self._audio_device is not None:
                dev_info = sd.query_devices(self._audio_device)
                print(f"[SpeechInput] Auto-selected microphone: [{self._audio_device}] {dev_info['name']}")

        if self._audio_device is None:
            print("[SpeechInput] No working microphone found!")
            return False

        self._sample_rate = native_rate
        print(f"[SpeechInput] Using device {self._audio_device} @ {native_rate}Hz "
              f"(resample to {self._whisper_rate}Hz for Whisper)")

        if self._wake_word:
            print(f"[SpeechInput] Wake word: '{self._wake_word}'")
        else:
            print("[SpeechInput] Wake word: disabled")

        self._backend_ready = True
        return True

    # ------------------------------------------------------------------
    # Internal — Recording Methods
    # ------------------------------------------------------------------
    def _record_with_vad(self):
        """
        Record audio using VAD to detect speech start/stop.
        Uses continuous InputStream and queue to prevent hardware latency/clicks.
        """
        import sounddevice as sd
        import queue

        fs = self._sample_rate
        frame_size = int(fs * self._frame_duration_ms / 1000)
        silence_frames_required = int(
            self._silence_timeout * 1000 / self._frame_duration_ms
        )

        # webrtcvad only supports 8000, 16000, 32000, 48000 Hz
        _SUPPORTED_VAD_RATES = {8000, 16000, 32000, 48000}
        vad_rate = fs if fs in _SUPPORTED_VAD_RATES else 16000

        recording = []
        silence_counter = 0
        speech_started = False

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[SpeechInput] InputStream status: {status}")
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=fs,
                blocksize=frame_size,
                device=self._audio_device,
                channels=1,
                dtype="int16",
                callback=audio_callback
            ):
                while self._running and self._enabled:
                    try:
                        frame = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # TTS feedback suppression: while Simon is speaking and barge-in
                    # is disabled, discard accumulated frames and sleep briefly.
                    # Uses a threading.Event set/cleared by VoiceEngine — thread-safe.
                    if (not self._barge_in) and self._tts_speaking.is_set():
                        while True:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        time.sleep(0.01)
                        continue

                    # Resample for VAD if mic rate is unsupported
                    if vad_rate != fs:
                        vad_frame = self._resample(frame.flatten(), fs, vad_rate)
                        vad_bytes = vad_frame.astype(np.int16).tobytes()
                    else:
                        vad_bytes = frame.tobytes()

                    is_speech = self._vad.is_speech(vad_bytes, vad_rate)

                    if is_speech:
                        speech_started = True
                        silence_counter = 0
                        recording.append(frame)
                    elif speech_started:
                        silence_counter += 1
                        recording.append(frame)

                        if silence_counter >= silence_frames_required:
                            break

                    # Safety limit
                    if len(recording) * self._frame_duration_ms / 1000 >= self._max_duration:
                        break

        except Exception as e:
            print(f"[SpeechInput] InputStream error: {e}")
            return None

        if not recording:
            return None

        return np.concatenate(recording, axis=0)

    def _record_timed(self, duration=4):
        """
        Record audio for a fixed duration (fallback when VAD unavailable).
        """
        import sounddevice as sd

        fs = self._sample_rate
        print("[SpeechInput] Listening...")
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="int16",
            device=self._audio_device,
        )
        sd.wait()
        return recording

    # ------------------------------------------------------------------
    # Internal — Silence Detection (RMS Energy)
    # ------------------------------------------------------------------
    def _is_silent(self, audio):
        """
        Determine if audio is silence using RMS energy analysis.
        More reliable than simple max-volume threshold.

        Returns:
            True if audio is below the silence threshold (no real speech).
        """
        if audio is None or len(audio) == 0:
            return True
        audio_float = audio.astype(np.float64)
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms < self._silence_rms_threshold

    # ------------------------------------------------------------------
    # Internal — Transcription (Direct In-Memory, No Disk I/O)
    # ------------------------------------------------------------------
    def _transcribe(self, audio):
        """
        Transcribe audio using Whisper — directly from NumPy array.
        No temporary file is created; audio is passed in-memory for
        faster transcription and lower latency.
        """
        try:
            # Resample to 16kHz if recorded at a different rate
            audio_16k = self._resample(audio, self._sample_rate, self._whisper_rate)

            # Convert int16 -> float32 normalized [-1.0, 1.0] for Whisper
            audio_float = audio_16k.flatten().astype(np.float32) / 32768.0

            result = self._model.transcribe(
                audio_float,
                language=self._language,
                fp16=False,  # CPU-safe
            )
            text = result["text"].strip()
            if not text:
                return ""
            return text
        except Exception as e:
            print(f"[SpeechInput] Transcription error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Internal — Command Parsing (Flexible + Wake Word)
    # ------------------------------------------------------------------
    def _parse_command(self, text):
        """
        Parse transcribed text into a SIMON command.

        NLP pipeline:
          1. Whisper hallucination filter
          2. Wake word check + removal
          3. Punctuation removal
          4. Verb tense normalization (saved→save, reading→read, etc.)
          5. Filler word removal (the, a, please, etc.)
          6. Flexible trigger matching

        Returns:
            dict: {"action": str, "args": str} or None
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # 1. Strip common Whisper hallucination patterns
        whisper_noise = [
            "thank you.", "thanks for watching.", "you", "bye.",
            "the end.", "...", "i'm going to",
        ]
        if text_lower in whisper_noise:
            return None

        # 2. Wake word check
        if self._wake_word:
            if self._wake_word not in text_lower:
                print(f"[SpeechInput] Wake word not detected: {text}")
                return None
            # Remove wake word
            text_lower = text_lower.replace(self._wake_word, "", 1).strip(" ,.")

        if not text_lower:
            return None

        # 3. Remove punctuation
        import re
        text_lower = re.sub(r"[^\w\s]", "", text_lower)

        # 4. Normalize verb tenses (Whisper often transcribes past/present)
        verb_map = {
            "saved": "save",
            "saves": "save",
            "saving": "save",
            "reads": "read",
            "reading": "read",
            "navigating": "navigate",
            "navigated": "navigate",
            "navigates": "navigate",
            "going": "go",
            "goes": "go",
            "taking": "take",
            "stopped": "stop",
            "stopping": "stop",
            "stops": "stop",
            "exiting": "exit",
            "exited": "exit",
            "cancelled": "cancel",
            "cancelling": "cancel",
            "remembered": "remember",
            "remembering": "remember",
            "scanned": "scan",
            "scanning": "scan",
            "switched": "switch",
            "switching": "switch",
        }
        for wrong, correct in verb_map.items():
            text_lower = re.sub(r'\b' + wrong + r'\b', correct, text_lower)

        # 5. Remove filler words (whole words only)
        filler_words = ["the", "a", "an", "please", "can", "you",
                        "could", "hey", "okay", "this", "that", "my"]
        words = text_lower.split()
        words = [w for w in words if w not in filler_words]
        text_lower = " ".join(words)

        # 6. Flexible trigger matching
        for trigger, action in sorted(COMMAND_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if trigger in text_lower:
                args = text_lower.replace(trigger, "", 1).strip()
                # Strip any residual wake word from args so e.g. "Simon navigate"
                # doesn't set "simon" as the destination after wake-word removal.
                if self._wake_word and args == self._wake_word:
                    args = ""
                return {"action": action, "args": args}

        # No recognized command
        print(f'[SpeechInput] Not a known command: "{text}"')
        return None

    # ------------------------------------------------------------------
    # Internal — Authentication Hook
    # ------------------------------------------------------------------
    def _verify_owner(self, audio, raw_text):
        """
        Extensible authentication hook for speaker verification.
        Currently implements a lightweight vocal PIN/passphrase check.
        Can be extended later to use ML voice biometrics on the `audio` buffer.
        """
        pin = getattr(config, "VOICE_PIN", None)
        if pin:
            # Lightweight verification: check if PIN was spoken in the command
            if str(pin).lower() not in raw_text.lower():
                return False
        
        # Future: extract embedding from `audio` and compare to owner's voice profile
        return True

    # ------------------------------------------------------------------
    # Internal — Main Loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        """Background thread: init model, then listen for commands."""
        if not self._init_backend():
            print("[SpeechInput] Backend init failed — thread exiting.")
            return

        while self._running:
            if not self._enabled:
                time.sleep(0.3)
                continue

            try:
                # Record audio
                if self._vad:
                    audio = self._record_with_vad()
                else:
                    audio = self._record_timed()

                if audio is None or len(audio) == 0:
                    time.sleep(0.05)  # Reduce CPU spikes
                    continue

                # RMS-based silence detection
                if self._is_silent(audio):
                    time.sleep(0.05)
                    continue

                # Transcribe (direct in-memory — no disk I/O)
                text = self._transcribe(audio)
                if not text:
                    continue
                print(f"[SpeechInput] Heard: {text}")

                # Parse command (flexible matching + wake word)
                command = self._parse_command(text)
                if command:
                    # Security: State-changing commands must pass authentication
                    state_changing_actions = {"navigate", "cancel_nav", "save_face"}
                    if command["action"] in state_changing_actions:
                        if not self._verify_owner(audio, text):
                            print(f"[SpeechInput] Unauthorized command blocked: {command['action']}")
                            if self.voice:
                                self.voice.speak("Command unauthorized.", priority=2)
                            continue

                    try:
                        self._command_queue.put_nowait(command)
                        if self.voice:
                            self.voice.speak(f"Command: {command['action']}.", priority=3)
                    except queue.Full:
                        print("[SpeechInput] Command queue full. Dropping command.")

            except Exception as e:
                print(f"[SpeechInput] Error: {e}")
                time.sleep(1)

            # Small sleep between cycles to reduce CPU usage
            time.sleep(0.05)


# ======================================================================
#  Standalone Test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SIMON Speech Input — Standalone Test")
    print("  Say 'Simon, navigate to library' or similar")
    print("=" * 60)

    # Pre-download model in main thread (shows progress bar, no timeout)
    import whisper
    model_name = getattr(config, "WHISPER_MODEL", "small")
    print(f"\n[Test] Downloading/loading Whisper '{model_name}' model...")
    print("[Test] (First run downloads the model — this is a one-time wait)\n")
    whisper.load_model(model_name)
    print("[Test] Model ready!\n")

    speech = SpeechInput()
    speech.start()
    speech.enabled = True

    # Backend init is fast now (model is cached)
    print("[Test] Waiting for backend to initialize...")
    for _ in range(120):  # up to 60 seconds
        if speech._backend_ready:
            break
        time.sleep(0.5)

    if not speech._backend_ready:
        print("[Test] Backend not ready — check Whisper/sounddevice installation.")
    else:
        print("[Test] Listening for 20 seconds — speak a command!")
        start = time.time()
        while time.time() - start < 20:
            cmd = speech.get_command()
            if cmd:
                print(f"[Test] >>> COMMAND: {cmd}")
            time.sleep(0.5)

    speech.stop()
    print("=== Test Complete ===")

