"""Base classes for processing stages."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from ray_curator.data import Task

T = TypeVar("T", bound=Task)


class StageType(Enum):
    """Types of processing stages."""

    READER = "reader"
    DOWNLOADER = "downloader"
    EXTRACTOR = "extractor"
    FILTER = "filter"
    TRANSFORMER = "transformer"
    WRITER = "writer"
    FUSED = "fused"
    CLASSIFIER = "classifier"


class ProcessingStage(ABC, Generic[T]):
    """Base class for all processing stages.
    Processing stages operate on Task objects (or subclasses like DocumentBatch).
    Each stage type can declare what type of Task it processes.
    Stages can return either:
    - A single task (typical for transformations)
    - A list of tasks (for stages that split work, like readers)
    - None (for filtered out tasks)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this stage."""

    @property
    @abstractmethod
    def stage_type(self) -> StageType:
        """Type of stage."""

    @property
    def can_fuse_with(self) -> list[StageType]:
        """List of stage types this stage can be fused with."""
        return []

    @property
    def requires_gpu(self) -> bool:
        """Whether this stage requires GPU resources."""
        return False

    @property
    def gpu_memory_gb(self) -> float:
        """GPU memory required in GB (if requires_gpu is True)."""
        return 0.0

    @property
    def cpu_cores(self) -> float:
        """Number of CPU cores required."""
        return 1.0

    @abstractmethod
    def process(self, task: T) -> None | T | list[T]:
        """Process a task and return the result.
        Args:
            task: Input task to process
        Returns:
            - Single task: For 1-to-1 transformations
            - List of tasks: For 1-to-many transformations (e.g., readers)
            - None: If the task should be filtered out
        """

    def validate_input(self, task: T) -> bool:
        """Validate input task meets requirements.
        Args:
            task: Task to validate
        Returns:
            True if valid, False otherwise
        """
        return True

    def setup(self) -> None:
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        """

    def teardown(self) -> None:
        """Teardown method called once after processing ends.
        Override this method to perform any cleanup.
        """

    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage.
        Returns:
            Dictionary of configuration parameters
        """
        return {
            "name": self.name,
            "stage_type": self.stage_type.value,
            "requires_gpu": self.requires_gpu,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_cores": self.cpu_cores,
        }
