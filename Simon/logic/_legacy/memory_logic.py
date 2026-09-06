# logic/memory_logic.py
import time

class MemoryLogic:
    MAX_MEMORY_ENTRIES = 200       # Hard cap on tracked objects
    MEMORY_EXPIRE_SECONDS = 30.0  # Auto-expire entries older than this
    _CLEANUP_INTERVAL = 50        # Run cleanup every N calls to should_speak

    def __init__(self, cooldown=5.0):
        """
        cooldown: minimum time (seconds) before repeating
        the same object description
        """
        self.cooldown = cooldown
        self.last_spoken = {}
        self._call_count = 0

    def should_speak(self, det):
        """
        Returns True if the object should be spoken now,
        False otherwise.
        """
        self._call_count += 1

        # Periodic cleanup to prevent unbounded growth
        if self._call_count % self._CLEANUP_INTERVAL == 0:
            self._cleanup()

        # Unique key and cooldown per detection type
        if det.cls == "face":
            key = f"face_{det.face_id}"
            cooldown = 10.0
        elif det.cls == "text":
            key = f"text_{det.text}"
            cooldown = 12.0
        else:
            key = f"{det.cls}_{det.position}_{det.distance}"
            cooldown = self.cooldown

        current_time = time.time()

        # First time seeing this object
        if key not in self.last_spoken:
            self.last_spoken[key] = current_time
            return True

        # Check cooldown
        if current_time - self.last_spoken[key] >= cooldown:
            self.last_spoken[key] = current_time
            return True

        return False

    def _cleanup(self):
        """
        Remove stale entries to prevent unbounded memory growth.
        Two-phase: expire old entries, then enforce hard cap.
        """
        now = time.time()

        # Phase 1: remove entries older than MEMORY_EXPIRE_SECONDS
        self.last_spoken = {
            k: t for k, t in self.last_spoken.items()
            if now - t < self.MEMORY_EXPIRE_SECONDS
        }

        # Phase 2: enforce hard cap (evict oldest if still over limit)
        if len(self.last_spoken) > self.MAX_MEMORY_ENTRIES:
            sorted_items = sorted(self.last_spoken.items(), key=lambda x: x[1])
            self.last_spoken = dict(sorted_items[-self.MAX_MEMORY_ENTRIES:])

    def reset(self):
        """Clear memory when scene changes drastically"""
        self.last_spoken.clear()
        self._call_count = 0
