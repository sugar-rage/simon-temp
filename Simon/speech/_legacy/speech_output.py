"""
SIMON Speech Output — Thread-safe pyttsx3-based offline text-to-speech.

Uses a dedicated speech thread with a priority queue to prevent
RuntimeError: 'run loop already started' when multiple modules
(object detection, OCR, voice commands) call speak() simultaneously.

Features:
  - Thread-safe: any thread can call speak() safely
  - Priority queue: higher priority messages go first
  - Speech cooldown: prevents repeating the same alert within N seconds
  - Fully offline, no model downloads required

Usage:
    from speech.speech_output import VoiceEngine
    tts = VoiceEngine()
    tts.speak("Hello from Simon")
    tts.speak("Alert!", priority=3)
"""

import pyttsx3
import threading
import queue
import time


class VoiceEngine:
    """Thread-safe offline TTS using pyttsx3 with speech queue and cooldown."""

    def __init__(self, rate=165, cooldown=3.0):
        """
        Initialize the voice engine.

        Args:
            rate:     Speech rate (words per minute).
            cooldown: Minimum seconds before repeating the same message.
        """
        self._rate = rate
        self._cooldown = cooldown

        # Speech queue: (priority, timestamp, text)
        # Lower priority number = higher priority
        self._queue = queue.PriorityQueue()

        # Cooldown tracking: {message_text: last_spoken_time}
        self._last_spoken = {}
        self._speak_count = 0
        self._MAX_COOLDOWN_ENTRIES = 200  # Hard cap on cooldown cache size
        self._lock = threading.Lock()

        # Speaking-state signal: set while runAndWait() is active so that
        # SpeechInput can pause microphone listening to prevent feedback.
        # threading.Event is thread-safe; no extra lock needed.
        self._speaking = threading.Event()

        # Optional back-reference to SpeechInput for barge-in control.
        self._speech_input = None

        # Dedicated speech thread
        self._running = True
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API — speaking state
    # ------------------------------------------------------------------
    @property
    def is_speaking(self):
        """True while TTS audio is actively playing."""
        return self._speaking.is_set()

    def register_speech_input(self, stt):
        """
        Wire a SpeechInput instance for automatic barge-in control.

        VoiceEngine will call stt.pause_for_tts() before each utterance
        and stt.resume_after_tts() when it finishes, so that the
        microphone is silenced while Simon is speaking.

        Args:
            stt: SpeechInput instance.
        """
        self._speech_input = stt

    def speak(self, text, priority=5):
        """
        Queue text to be spoken. Thread-safe, can be called from any thread.

        Args:
            text:     String to speak.
            priority: 1 (highest) to 10 (lowest). Default 5.
        """
        if not text:
            return

        # Check cooldown — skip if same message was spoken recently
        now = time.time()

        with self._lock:
            last_time = self._last_spoken.get(text, 0)
            if now - last_time < self._cooldown:
                return

            self._last_spoken[text] = now
            self._speak_count += 1
            if self._speak_count % 50 == 0:
                expire_threshold = now - (2 * self._cooldown)
                self._last_spoken = {
                    k: t for k, t in self._last_spoken.items()
                    if t > expire_threshold
                }
                # Hard cap: evict oldest if still over limit
                if len(self._last_spoken) > self._MAX_COOLDOWN_ENTRIES:
                    sorted_items = sorted(self._last_spoken.items(), key=lambda x: x[1])
                    self._last_spoken = dict(sorted_items[-self._MAX_COOLDOWN_ENTRIES:])

        self._queue.put((priority, now, text))

    def stop(self):
        """Stop the speech thread."""
        self._running = False
        # Put a sentinel to unblock the queue
        self._queue.put((0, 0, None))
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def _speech_loop(self):
        """Dedicated thread: processes speech queue one at a time."""
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)

        self._current_priority = 10

        def on_word(name, location, length):
            """
            pyttsx3 callback executed during speech (runs on this same thread).
            This is the only safe place to interrupt pyttsx3 without COM deadlocks.
            """
            if not self._running:
                engine.stop()
                return

            # Check if a higher priority message (like a hazard) is waiting
            with self._queue.mutex:
                if self._queue.queue:
                    # PriorityQueue's underlying list has the highest priority item at index 0
                    next_priority = self._queue.queue[0][0]
                    if next_priority < self._current_priority:
                        # Abort current speech. runAndWait() will return immediately.
                        engine.stop()

        engine.connect('word', on_word)

        while self._running:
            try:
                priority, timestamp, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # Sentinel to stop
                break

            self._current_priority = priority

            try:
                print(f"SIMON: {text}")
                engine.say(text)

                # Signal that speaking is starting so SpeechInput can mute.
                self._speaking.set()
                if self._speech_input is not None:
                    self._speech_input.pause_for_tts()

                engine.runAndWait()

            except Exception as e:
                print(f"[VoiceEngine] TTS error: {e}")
                # Re-init engine if it crashed
                try:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self._rate)
                    engine.connect('word', on_word)
                except Exception:
                    pass
            finally:
                # Always clear speaking state and resume listening, even on error.
                self._speaking.clear()
                if self._speech_input is not None:
                    self._speech_input.resume_after_tts()

        try:
            engine.stop()
        except Exception:
            pass
