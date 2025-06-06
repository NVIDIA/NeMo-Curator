"""Base classes for processing stages."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Generic, List, Optional, TypeVar

from ray_curator.data import Task

T = TypeVar("T", bound=Task)

# Global registry for auto-discoverable stages.  The key is the *class name*
# (unique) and the value is the class object itself.

_STAGE_REGISTRY: Dict[str, Type["ProcessingStage"]] = {}


class StageMeta(ABCMeta):
    """Metaclass that automatically registers concrete Stage subclasses.

    A class is considered *concrete* if it directly inherits from
    :class:`ProcessingStage` **and** implements a ``name`` property.  Abstract
    helper classes (e.g. *ProcessingStage* itself) will not be added to the
    registry because they have the ``_is_abstract`` attribute set.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Skip registration for the abstract roots
        if namespace.get("_is_abstract_root", False):
            return cls

        # Only register subclasses that ultimately derive from ProcessingStage
        # but are not abstract.
        from inspect import isabstract  # local import to avoid cycle during class creation

        if "ProcessingStage" in [base.__name__ for base in cls.mro()[1:]] and not isabstract(cls):
            # Ensure no duplicate class names (helps when reloading in notebooks)
            _STAGE_REGISTRY[cls.__name__] = cls

        return cls


def get_stage_class(name: str) -> Type["ProcessingStage"]:
    """Retrieve a registered stage class by its *class name*.

    Raises
    ------
    KeyError
        If no stage with that name is registered.
    """

    return _STAGE_REGISTRY[name]


class StageType(Enum):
    """Types of processing stages."""

    READER = "reader"
    DOWNLOADER = "downloader"
    EXTRACTOR = "extractor"
    FILTER = "filter"
    TRANSFORMER = "transformer"
    WRITER = "writer"
    FUSED = "fused"
    COMPOSITE = "composite"  # High-level stages that decompose
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

    _is_abstract_root = True  # prevent base from registering itself

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

    def is_composite(self) -> bool:
        """Check if this is a composite stage that decomposes into sub-stages."""
        return self.stage_type == StageType.COMPOSITE

    def decompose(self) -> List["ProcessingStage"]:
        """Decompose this stage into execution stages.
        
        High-level composite stages should override this method to return
        the list of low-level stages they represent.
        
        Returns:
            List of execution stages. Returns [self] by default for regular stages.
        """
        return [self]

    def supports_batch_processing(self) -> bool:
        """Whether this stage supports vectorized batch processing.
        
        This is automatically determined by checking if the stage has
        overridden the process_batch method from the base class.
        """
        # Check if process_batch has been overridden
        return type(self).process_batch is not ProcessingStage.process_batch

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

    def process_batch(self, tasks: list[T]) -> list[None | T | list[T]]:
        """Process a batch of tasks and return results.
        
        Override this method to enable batch processing for your stage.
        If not overridden, the stage will only support single-task processing.
        
        Args:
            tasks: List of input tasks to process
            
        Returns:
            List of results, where each result can be:
            - Single task: For 1-to-1 transformations
            - List of tasks: For 1-to-many transformations
            - None: If the task should be filtered out
            
        Note: The returned list should have the same length as the input list,
        with each element corresponding to the result of processing the task
        at the same index.
        """
        # Default implementation: process tasks one by one
        # This is only used as a fallback if a stage doesn't override this method
        results = []
        for task in tasks:
            result = self.process(task)
            results.append(result)
        return results

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
            "supports_batch_processing": self.supports_batch_processing(),
        }


class CompositeStage(ProcessingStage[T], ABC):
    """Base class for high-level composite stages.
    
    Composite stages are user-facing stages that decompose into multiple
    low-level execution stages during pipeline planning. They provide a
    simplified API while maintaining fine-grained control at execution time.
    
    Composite stages never actually execute - they only exist to be decomposed
    into their constituent execution stages.
    """

    @property
    def stage_type(self) -> StageType:
        """Composite stages always have COMPOSITE type."""
        return StageType.COMPOSITE

    @abstractmethod
    def decompose(self) -> List[ProcessingStage]:
        """Decompose into execution stages.
        
        This method must be implemented by composite stages to define
        what low-level stages they represent.
        
        Returns:
            List of execution stages that will actually run
        """

    def process(self, task: T) -> None | T | list[T]:
        """Composite stages should never be executed directly."""
        raise RuntimeError(
            f"Composite stage '{self.name}' should not be executed directly. "
            "It should be decomposed into execution stages during planning."
        )

    def get_description(self) -> str:
        """Get a description of what this composite stage does.
        
        Override this to provide user-friendly documentation.
        """
        return f"Composite stage: {self.name}"
