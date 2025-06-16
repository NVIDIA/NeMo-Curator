import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

import ray_curator.stages.io.writer.utils as writer_utils
from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch, FileGroupTask


@dataclass
class DataFrameWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Stage that writes a DocumentBatch to files in various formats.

    This stage accepts DocumentBatch tasks and writes them to files in the specified format.
    Supported formats: 'jsonl', 'parquet'
    """

    output_dir: str
    format: Literal["jsonl", "parquet"] = "jsonl"
    writer_kwargs: dict[str, Any] = field(default_factory=dict)
    storage_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in ["jsonl", "parquet"]:
            msg = f"Invalid format: {self.format}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Setup filesystem once per worker."""
        from fsspec.utils import infer_storage_options

        storage_options_inferred = infer_storage_options(self.output_dir)
        self.fs = fsspec.filesystem(storage_options_inferred["protocol"], **self.storage_options)

    def process(self, task: DocumentBatch) -> FileGroupTask:
        """Process a DocumentBatch and write to files.

        Args:
            task (DocumentBatch): DocumentBatch containing data to write

        Returns:
            FileGroupTask: Task containing paths to written files
        """
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):  # noqa: SLF001
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        # Create output directory
        self.fs.makedirs(self.output_dir, exist_ok=True)

        # Generate filename with appropriate extension
        file_extension = "jsonl" if self.format == "jsonl" else "parquet"
        file_path = os.path.join(self.output_dir, f"{filename}.{file_extension}")

        # Skip if file already exists (idempotent writes)
        if self.fs.exists(file_path):
            logger.debug(f"File {file_path} already exists, skipping")
        else:
            self._write_data(task, file_path)

            # Get record count for logging
            record_count = len(task.data)
            logger.debug(f"Written {record_count} records to {file_path}")

        # Create FileGroupTask with written files
        return FileGroupTask(
            task_id=f"{filename}_written",  # TODO : Discuss how task_id should be handled
            dataset_name=task.dataset_name,
            data=[file_path],
            _metadata={
                **task._metadata,  # noqa: SLF001
                "output_dir": self.output_dir,
                "format": self.format,
            },
            _stage_perf=task._stage_perf,  # noqa: SLF001
        )

    def _write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to file based on format, using native data type when possible."""
        if isinstance(task.data, pd.DataFrame) or self.format == "jsonl":
            df = task.to_pandas()  # should be no-op if already pandas
            if self.format == "jsonl":
                df.to_json(
                    file_path,
                    orient="records",
                    lines=True,
                    storage_options=self.storage_options,
                    **self.writer_kwargs,
                )
            elif self.format == "parquet":
                df.to_parquet(
                    file_path,
                    storage_options=self.storage_options,
                    **self.writer_kwargs,
                )
        else:
            pq.write_table(task.data, file_path, filesystem=self.fs, **self.writer_kwargs)
