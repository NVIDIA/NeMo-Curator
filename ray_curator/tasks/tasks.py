"""Task data structures for the ray-curator pipeline framework."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union
from ray_curator.utils.performance_utils import StagePerfStats

import pandas as pd
import pyarrow as pa




@dataclass
class Task:
    """Abstract base class for tasks in the pipeline.
    A task represents a batch of data to be processed. Different modalities
    (text, audio, video) can implement their own task types.
    Attributes:
        task_id: Unique identifier for this task
        dataset_name: Name of the dataset this task belongs to
        dataframe_attribute: Name of the attribute that contains the dataframe data. We use this for input/output validations.
        _stage_perf: List of stages perfs this task has passed through
    """

    task_id: str
    dataset_name: str
    dataframe_attribute: str = "data"
    _stage_perf: list[StagePerfStats] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        """Get the number of items in this task."""
        return 0

    def add_stage_perf(self, perf_stats: StagePerfStats) -> None:
        """Add performance stats for a stage."""
        self._stage_perf.append(perf_stats)

@dataclass
class _EmptyTask(Task):
    """Dummy task for testing."""
    data: Any = None

    @property
    def num_items(self) -> int:
        """Get the number of items in this task."""
        return 0

    def validate(self) -> bool:
        """Validate the task data."""
        return True

# Empty tasks are just used for `ls` stages
EmptyTask = _EmptyTask(task_id="empty", dataset_name="empty", data=None)


@dataclass
class DocumentBatch(Task):
    """Task for processing batches of text documents.
    Documents are stored as a dataframe (PyArrow table or Pandas DataFrame).
    The schema is flexible - users can specify which columns contain text
    and other relevant data.
    Attributes:
        text_column: Name of the column containing text content
        id_column: Name of the column containing document IDs (optional)
        additional_columns: List of other columns to preserve during processing
    """

    text_column: str = "content"
    id_column: str | None = "id"
    additional_columns: list[str] = field(default_factory=list)
    data: Union[pa.Table, pd.DataFrame] = field(default_factory=pa.Table)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pyarrow(self) -> pa.Table:
        """Convert data to PyArrow table."""
        if isinstance(self.data, pa.Table):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data)
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to PyArrow table")

    def to_pandas(self) -> pd.DataFrame:
        """Convert data to Pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, pa.Table):
            return self.data.to_pandas()
        else:
            raise ValueError(f"Cannot convert {type(self.data)} to Pandas DataFrame")

    @property
    def num_items(self) -> int:
        """Get the number of documents in this batch."""
        return len(self.data)

@dataclass
class ImageObject:
    """Represents a single image with metadata."""

    image_path: str = ""
    image_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageBatch(Task):
    """Task for processing batches of images.
    Images are stored as a list of ImageObject instances, each containing
    the path to the image and associated metadata.
    """

    data: list[ImageObject] = field(default_factory=list)

