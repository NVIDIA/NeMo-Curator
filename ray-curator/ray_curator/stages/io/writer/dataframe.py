import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import fsspec
from loguru import logger

import ray_curator.stages.io.writer.utils as writer_utils
from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch, FileGroupTask


@dataclass
class BaseWriter(ProcessingStage[DocumentBatch, FileGroupTask], ABC):
    """Base class for all writer stages.

    This abstract base class provides common functionality for writing DocumentBatch
    tasks to files, including file naming, metadata handling, and filesystem operations.
    """

    output_dir: str
    file_extension: str
    storage_options: dict[str, Any] = field(default_factory=dict)
    _name: str = "BaseWriter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Setup filesystem once per worker."""
        from fsspec.utils import infer_storage_options

        storage_options_inferred = infer_storage_options(self.output_dir)
        self.fs = fsspec.filesystem(storage_options_inferred["protocol"], **self.storage_options)

    def get_file_extension(self) -> str:
        """Return the file extension for this writer format."""
        return self.file_extension

    @abstractmethod
    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to file using format-specific implementation."""

    def process(self, task: DocumentBatch) -> FileGroupTask:
        """Process a DocumentBatch and write to files.

        Args:
            task (DocumentBatch): DocumentBatch containing data to write

        Returns:
            FileGroupTask: Task containing paths to written files
        """
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        # Create output directory
        self.fs.makedirs(self.output_dir, exist_ok=True)

        # Generate filename with appropriate extension
        file_extension = self.get_file_extension()
        file_path = self.fs.sep.join([self.output_dir, f"{filename}.{file_extension}"])

        # Skip if file already exists (idempotent writes)
        if self.fs.exists(file_path):
            logger.debug(f"File {file_path} already exists, skipping")
        else:
            self.write_data(task, file_path)
            logger.debug(f"Written {task.num_items} records to {file_path}")

        # Create FileGroupTask with written files
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_path],
            _metadata={
                **task._metadata,
                "output_dir": self.output_dir,
                "format": self.get_file_extension(),
            },
            _stage_perf=task._stage_perf,
        )
