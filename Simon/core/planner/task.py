"""Task dataclass with lifecycle management.

A Task represents an executable unit of work — either a top-level
command or a subtask within a decomposed plan.  Tasks follow a
lifecycle: PENDING → RUNNING → COMPLETED | FAILED | CANCELLED.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from core.models.enums import TaskStatus


class TaskType(Enum):
    """Types of tasks the planner can decompose."""

    NAVIGATE = "navigate"
    READ_TEXT = "read_text"
    SAVE_FACE = "save_face"
    DESCRIBE_SCENE = "describe_scene"
    REPORT_STATUS = "status"
    STOP = "stop"
    CANCEL_NAV = "cancel_nav"
    SPEAK = "speak"
    CUSTOM = "custom"


@dataclass
class Task:
    """An executable task with lifecycle tracking.

    Attributes
    ----------
    task_id : str
        Unique identifier (auto-generated UUID).
    task_type : TaskType
        What kind of work this task represents.
    args : dict[str, Any]
        Task-specific arguments.
    status : TaskStatus
        Current lifecycle state.
    subtasks : list[Task]
        Ordered child tasks (for decomposed commands).
    parent_id : str or None
        Parent task ID if this is a subtask.
    priority : int
        Execution priority (lower = more urgent).
    result : dict or None
        Result data after completion.
    error : str or None
        Error message if failed.
    created_at : float
        Creation timestamp.
    started_at : float or None
        When execution began.
    completed_at : float or None
        When the task finished (success or failure).
    """

    task_type: TaskType
    args: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list["Task"] = field(default_factory=list)
    parent_id: Optional[str] = None
    priority: int = 6
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def start(self) -> None:
        """Mark task as RUNNING."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Optional[dict[str, Any]] = None) -> None:
        """Mark task as COMPLETED."""
        self.status = TaskStatus.COMPLETED
        self.result = result or {}
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        """Mark task as FAILED."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    def cancel(self) -> None:
        """Mark task as CANCELLED."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()

    @property
    def is_active(self) -> bool:
        """True if the task is pending or running."""
        return self.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

    @property
    def is_terminal(self) -> bool:
        """True if the task has finished (success, failure, or cancellation)."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def duration_ms(self) -> Optional[float]:
        """Elapsed execution time in milliseconds, or None if not started."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000.0
