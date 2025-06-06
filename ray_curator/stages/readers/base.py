"""Base classes for readers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ray_curator.data import Task


@dataclass
class FileGroupTask(Task[list[str]]):
    """Task representing a group of files to be read.

    This is created during the planning phase and passed to reader stages.
    """

    reader_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate that we have files to read."""
        return isinstance(self.data, list) and len(self.data) > 0

    @property
    def num_items(self) -> int:
        """Number of files in this group."""
        return len(self.data)

    @property
    def file_paths(self) -> list[str]:
        """Get the file paths."""
        return self.data


class Reader(ABC):
    """Abstract base class for data readers.

    Readers are configuration objects that specify how to read data.
    During pipeline planning, they create FileGroupTasks that are
    processed by the actual reading stages.
    """

    @abstractmethod
    def create_file_groups(self) -> list[FileGroupTask]:
        """Create file group tasks based on partitioning strategy.

        This is called during pipeline planning, not during execution.

        Returns:
            List of FileGroupTask objects
        """

    @abstractmethod
    def get_reader_stage_name(self) -> str:
        """Get the name of the stage that will process these file groups.

        Returns:
            Name of the processing stage
        """

    @abstractmethod
    def get_stage_config(self) -> dict[str, Any]:
        """Get configuration to pass to the processing stage.

        Returns:
            Configuration dictionary for the stage
        """
