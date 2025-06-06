"""Base classes for processing stages."""

from __future__ import annotations

from abc import ABC, abstractmethod, ABCMeta
from enum import Enum
from typing import Any, Generic, List, Optional, TypeVar, Dict, Type
from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Task
from ray_curator.tasks.utils import get_columns
from loguru import logger

X = TypeVar("X", bound=Task)  # Input task type
Y = TypeVar("Y", bound=Task)  # Output task type

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


class ProcessingStage(ABC, Generic[X, Y], metaclass=StageMeta):
    """Base class for all processing stages.
    Processing stages operate on Task objects (or subclasses like DocumentBatch).
    Each stage type can declare what type of Task it processes as input (X)
    and what type it produces as output (Y).
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
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=1.0)

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return 1

    @property
    def num_workers(self) -> int | None:
        """Number of workers required. If None, then executor will determine the number of workers."""
        return None

    def validate_input(self, task: Task) -> bool:
        """Validate input task meets requirements.
        Args:
            task: Task to validate
        Returns:
            True if valid, False otherwise
        """
        required_attrs, required_columns = self.inputs()

        # Check required attributes exist
        for attr in required_attrs:
            if not hasattr(task, attr):
                logger.error(f"Task {task.task_id} missing required attribute: {attr}")
                return False

        # Check required columns exist
        if required_columns:
            task_columns = getattr(task, task.dataframe_attribute)
            task_columns = get_columns(task_columns)
            for col in required_columns:
                if col not in task_columns:
                    logger.error(f"Task {task.task_id} missing required column: {col}; required columns: {required_columns}; task columns: {task_columns}")
                    return False

        return True

    @abstractmethod
    def process(self, task: X) -> Y | list[Y]:
        """Process a task and return the result.
        Args:
            task: Input task to process
        Returns:
            - Single task: For 1-to-1 transformations
            - List of tasks: For 1-to-many transformations (e.g., readers)
            - None: If the task should be filtered out
        """

    def process_batch(self, tasks: list[X]) -> list[Y]:
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
            if not self.validate_input(task):
                msg = f"Task {task} failed validation for stage {self}"
                raise ValueError(msg)

            result = self.process(task)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once per node in distributed settings.
        Override this method to perform node-level initialization.
        Args:
            node_info: Information about the node (provided by some backends)
            worker_metadata: Information about the worker (provided by some backends)
        """

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata: Information about the worker (provided by some backends)
        """

    def teardown(self) -> None:
        """Teardown method called once after processing ends.
        Override this method to perform any cleanup.
        """

    def supports_batch_processing(self) -> bool:
        """Whether this stage supports vectorized batch processing.
        This is automatically determined by checking if the stage has
        overridden the process_batch method from the base class.
        """
        # Check if process_batch has been overridden
        return type(self).process_batch is not ProcessingStage.process_batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    

    @abstractmethod
    def inputs(self) -> tuple[list[str], list[str]]:
        """Define stage input requirements.
        
        Returns:
            Tuple of (required_attributes, required_columns) where:
            - required_attributes: List of task attributes that must be present
            - required_columns: List of dataframe columns that must be present
        """

    @abstractmethod  
    def outputs(self) -> tuple[list[str], list[str]]:
        """Define stage output specification.
        
        Returns:
            Tuple of (output_attributes, output_columns) where:
            - output_attributes: List of task attributes this stage adds/modifies
            - output_columns: List of dataframe columns this stage adds/modifies
        """

    def xenna_stage_spec(self) -> dict[str, Any]:
        """Get Xenna configuration for this stage.
        
        Returns:
            Dictionary containing Xenna-specific configuration
        """
        return {}
    
    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage."""
        return {
            "name": self.name,
            "resources": self.resources,
            "batch_size": self.batch_size,
            "supports_batch_processing": self.supports_batch_processing(),
        }
    

class CompositeStage(ProcessingStage[X, Y], ABC):
    """Base class for high-level composite stages.
    
    Composite stages are user-facing stages that decompose into multiple
    low-level execution stages during pipeline planning. They provide a
    simplified API while maintaining fine-grained control at execution time.
    
    Composite stages never actually execute - they only exist to be decomposed
    into their constituent execution stages.
    """

    def inputs(self) -> tuple[list[str], list[str]]:
        """Get the inputs for this stage."""
        return self.decompose()[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        """Get the outputs for this stage."""
        return self.decompose()[-1].outputs()
    
    @abstractmethod
    def decompose(self) -> List[ProcessingStage]:
        """Decompose into execution stages.
        
        This method must be implemented by composite stages to define
        what low-level stages they represent.
        
        Returns:
            List of execution stages that will actually run
        """

    def process(self, task: X) -> Y | list[Y]:
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