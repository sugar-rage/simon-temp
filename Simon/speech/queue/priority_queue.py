"""
Priority queue for speech synthesis tasks, allowing preemption.

Lower numeric values for SpeechPriority represent higher urgency.
This queue ensures that emergency warnings interrupt background info.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Callable, Optional

from speech.models.speech_priority import SpeechPriority


@dataclass(order=True)
class SpeakItem:
    """An item to be spoken, managed by the SpeechPriorityQueue."""
    
    # Priority is the primary sort key (lower = higher priority)
    priority: int
    
    # Timestamp is the secondary sort key (FIFO for same priority)
    timestamp: float
    
    # Counter for stable sorting when timestamps are identical
    _counter: int
    
    # The actual payload (not compared for sorting)
    text: str = field(compare=False)
    
    # Optional callback when playback completes or is aborted
    on_complete: Optional[Callable[[bool], None]] = field(default=None, compare=False)
    
    # Track if this item was aborted due to preemption
    aborted: bool = field(default=False, compare=False)


class SpeechPriorityQueue:
    """A thread-safe priority queue for speech items with preemption support."""

    def __init__(self):
        self._queue: PriorityQueue[SpeakItem] = PriorityQueue()
        self._counter = itertools.count()
        self._lock = threading.Lock()
        
        # Track the currently processing item to handle preemption
        self._current_item: Optional[SpeakItem] = None
        
        # Event to signal the player thread to abort current playback
        self.abort_event = threading.Event()

    def put(self, text: str, priority: SpeechPriority, on_complete: Optional[Callable[[bool], None]] = None) -> SpeakItem:
        """Enqueue a new speech item and handle potential preemption."""
        item = SpeakItem(
            priority=priority.value,
            timestamp=time.time(),
            _counter=next(self._counter),
            text=text,
            on_complete=on_complete
        )
        
        with self._lock:
            # Check if we need to preempt the currently playing item
            if self._current_item and not self._current_item.aborted:
                if priority.value < self._current_item.priority:
                    # New item is higher priority than current (lower value)
                    self._current_item.aborted = True
                    self.abort_event.set()
            
            self._queue.put(item)
            
        return item

    def get(self, timeout: Optional[float] = None) -> SpeakItem:
        """Get the next highest-priority item from the queue."""
        # This will block until an item is available
        item = self._queue.get(timeout=timeout)
        
        with self._lock:
            self._current_item = item
            self.abort_event.clear()
            
        return item

    def task_done(self) -> None:
        """Mark the current item as done processing."""
        with self._lock:
            if self._current_item:
                # Trigger callback if present
                if self._current_item.on_complete:
                    try:
                        # Pass True if it finished normally, False if aborted
                        self._current_item.on_complete(not self._current_item.aborted)
                    except Exception:
                        pass
                
                self._current_item = None
                self.abort_event.clear()
                
        self._queue.task_done()

    def clear_queue(self) -> None:
        """Clear all pending items in the queue (does not abort current)."""
        with self._lock:
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    item.aborted = True
                    if item.on_complete:
                        try:
                            item.on_complete(False)
                        except Exception:
                            pass
                    self._queue.task_done()
                except Empty:
                    break

    def abort_all(self) -> None:
        """Clear the queue AND abort the currently playing item."""
        with self._lock:
            if self._current_item:
                self._current_item.aborted = True
                self.abort_event.set()
        self.clear_queue()

    @property
    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()
