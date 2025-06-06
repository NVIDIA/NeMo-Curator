"""Base classes for readers."""

from dataclasses import dataclass, field
from typing import Any

from ray_curator.tasks import Task


@dataclass
class FileGroupTask(Task):
    """Task representing a group of files to be read.
    This is created during the planning phase and passed to reader stages.
    """

    reader_config: dict[str, Any] = field(default_factory=dict)
    data: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    file_paths: list[str] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        """Number of files in this group."""
        return len(self.data)

