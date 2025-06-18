"""Task data structures for the ray-curator pipeline framework."""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar

from ray_curator.utils.performance_utils import StagePerfStats

from .utils import _convert_numpy_to_native

T = TypeVar("T")

_TASK_REGISTRY: dict[str, type] = {}


class TaskMeta(ABCMeta):
    """Metaclass that automatically registers concrete Task subclasses.
    A class is considered *concrete* if it ultimately derives from
    :class:`Task` **and** is not abstract. Abstract helper classes
    (e.g. *Task* itself) will not be added to the registry.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: ANN001
        # Create the class first
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip registration for the abstract roots
        if namespace.get("_is_abstract_root", False):
            return cls

        # Only register subclasses that ultimately derive from Task
        # but are not abstract.
        from inspect import isabstract  # local import to avoid cycle during class creation

        if "Task" in [base.__name__ for base in cls.mro()[1:]] and not isabstract(cls):
            # Ensure no duplicate class names (helps when reloading in notebooks)
            _TASK_REGISTRY[cls.__name__] = cls

        return cls


def get_task_class(name: str) -> type["Task"]:
    """Retrieve a registered task class by its *class name*.

    Raises:
        KeyError: If no task with that name is registered.

    Returns:
        Task: The registered task class.
    """

    if name in _TASK_REGISTRY:
        return _TASK_REGISTRY[name]

    available_tasks = list(_TASK_REGISTRY.keys())
    msg = f"Unknown task type: {name}. Available tasks: {available_tasks}"
    raise KeyError(msg)


@dataclass
class Task(ABC, Generic[T], metaclass=TaskMeta):
    """Abstract base class for tasks in the pipeline.
    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.
    Attributes:
        task_id: Unique identifier for this task
        dataset_name: Name of the dataset this task belongs to
        dataframe_attribute: Name of the attribute that contains the dataframe data. We use this for input/output validations.
        _stage_perf: List of stages perfs this task has passed through
    """

    _is_abstract_root = True  # Prevent registration of the base Task class

    task_id: str
    dataset_name: str
    data: T
    _stage_perf: list[StagePerfStats] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self.validate()

    @property
    @abstractmethod
    def num_items(self) -> int:
        """Get the number of items in this task."""

    def add_stage_perf(self, perf_stats: StagePerfStats) -> None:
        """Add performance stats for a stage."""
        self._stage_perf.append(perf_stats)

    def __repr__(self) -> str:
        subclass_name = self.__class__.__name__
        return f"{subclass_name}(task_id={self.task_id}, dataset_name={self.dataset_name})"

    @abstractmethod
    def validate(self) -> bool:
        """Validate the task data."""

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for serialization.
        Uses dataclasses.asdict() to automatically serialize all fields,
        so subclasses don't need to override this method.
        """
        task_dict = asdict(self)
        # Add the task type for reconstruction
        task_dict["task_type"] = self.__class__.__name__
        return task_dict

    @classmethod
    def from_dict(cls, task_dict: dict[str, Any]) -> "Task":
        """Create task from dictionary.

        Args:
            task_dict: Dictionary with task_type, data, and stage_history

        Returns:
            Task instance
        """
        task_type = task_dict.pop("task_type")
        # Convert numpy arrays back to Python native types throughout the entire dict
        task_dict = _convert_numpy_to_native(task_dict)

        # Use the registry lookup
        task_class = get_task_class(task_type)
        return task_class(**task_dict)


@dataclass
class _EmptyTask(Task[None]):
    """Dummy task for testing."""

    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        """Validate the task data."""
        return True


# Empty tasks are just used for `ls` stages
EmptyTask = _EmptyTask(task_id="empty", dataset_name="empty", data=None)
