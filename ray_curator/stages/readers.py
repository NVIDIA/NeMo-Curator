"""Reader stages that process FileGroupTasks created during planning."""

import logging
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.json as pj

from ray_curator.data import DocumentBatch
from ray_curator.readers.base import FileGroupTask

from .base import ProcessingStage, StageType

logger = logging.getLogger(__name__)


class JsonlProcessingStage(ProcessingStage[FileGroupTask]):
    """Stage that processes a group of JSONL files into a DocumentBatch.
    This stage is used internally by the pipeline to process file groups
    created by JsonlReader during the planning phase.
    """

    def __init__(self):
        """Initialize the JSONL processing stage."""

    @property
    def name(self) -> str:
        return "jsonl_processing_stage"

    @property
    def stage_type(self) -> StageType:
        return StageType.READER

    def process(self, task: FileGroupTask) -> DocumentBatch | None:
        """Process a single group of JSONL files.
        Args:
            task: FileGroupTask containing file paths and configuration
        Returns:
            DocumentBatch with the data from these files
        """
        if not task.file_paths:
            logger.warning(f"No files to process in task {task.task_id}")
            return None

        # Extract configuration from the task
        config = task.reader_config
        text_column = config.get("text_column", "content")
        id_column = config.get("id_column", "id")
        additional_columns = config.get("additional_columns", [])
        reader = config.get("reader", "pandas")
        reader_kwargs = config.get("reader_kwargs", {})
        storage_options = config.get("storage_options", {})

        # Read the files
        if reader == "pandas":
            df = self._read_with_pandas(task.file_paths, storage_options, reader_kwargs)
        elif reader == "pyarrow":
            df = self._read_with_pyarrow(task.file_paths, reader_kwargs)
        else:
            raise ValueError(f"Unknown reader: {reader}")

        if df is None or (hasattr(df, "empty") and df.empty):
            logger.warning(f"No data read from files in task {task.task_id}")
            return None

        # Create DocumentBatch
        batch = DocumentBatch(
            task_id=f"{task.task_id}_processed",
            dataset_name=task.dataset_name,
            data=df,
            metadata={
                **task.metadata,
                "source_files": task.file_paths,
                "num_files": len(task.file_paths),
                "reader": self.name,
            },
            text_column=text_column,
            id_column=id_column,
            additional_columns=additional_columns,
        )
        batch.add_stage(self.name)

        return batch

    def _read_with_pandas(
        self, file_paths: list[str], storage_options: dict[str, Any], reader_kwargs: dict[str, Any]
    ) -> pd.DataFrame:
        """Read JSONL files using pandas."""
        dfs = []

        for file_path in file_paths:
            try:
                df = pd.read_json(file_path, lines=True, storage_options=storage_options, **reader_kwargs)
                dfs.append(df)
                logger.debug(f"Read {len(df)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        # Concatenate all dataframes
        return pd.concat(dfs, ignore_index=True)

    def _read_with_pyarrow(self, file_paths: list[str], reader_kwargs: dict[str, Any]) -> pa.Table:
        """Read JSONL files using pyarrow."""
        tables = []

        for file_path in file_paths:
            try:
                table = pj.read_json(file_path, **reader_kwargs)
                tables.append(table)
                logger.debug(f"Read {len(table)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not tables:
            # Return empty table
            return pa.table({})

        # Concatenate all tables
        return pa.concat_tables(tables)
