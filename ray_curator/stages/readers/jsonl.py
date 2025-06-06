"""JSONL reader composite stage."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ray_curator.data import Task, DocumentBatch
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.readers.base import FileGroupTask
from ray_curator.utils.file_utils import get_all_files_paths_under

logger = logging.getLogger(__name__)


@dataclass
class FilePartitioningStage(ProcessingStage[Task]):
    """Stage that partitions input file paths into FileGroupTasks.
    
    This stage runs as a dedicated processing stage (not on the driver)
    and creates file groups based on the partitioning strategy.
    """
    
    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] | None = None
    storage_options: dict[str, Any] | None = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.file_extensions is None:
            self.file_extensions = [".jsonl", ".json"]
        if self.storage_options is None:
            self.storage_options = {}
    
    @property
    def name(self) -> str:
        return "file_partitioning"
    
    @property
    def is_source_stage(self) -> bool:
        """This is a source stage that creates initial file group tasks."""
        return True
    
    def process(self, task: Task) -> list[FileGroupTask] | None:
        """Process the initial task to create file group tasks.
        
        This stage expects a simple Task with file paths information
        and outputs multiple FileGroupTasks for parallel processing.
        """
        # Get list of files
        files = self._get_file_list()
        
        if not files:
            logger.warning(f"No files found matching pattern: {self.file_paths}")
            return None
        
        # Partition files
        if self.files_per_partition:
            partitions = self._partition_by_count(files, self.files_per_partition)
        elif self.blocksize:
            partitions = self._partition_by_size(files, self.blocksize)
        else:
            # All files in one group
            partitions = [files]
        
        # Create FileGroupTask for each partition
        tasks = []
        dataset_name = self._get_dataset_name(files)
        
        for i, file_group in enumerate(partitions):
            file_task = FileGroupTask(
                task_id=f"file_group_{i}",
                dataset_name=dataset_name,
                file_paths=file_group,
                metadata={
                    "partition_index": i,
                    "total_partitions": len(partitions),
                    "storage_options": self.storage_options,
                },
                reader_config={},  # Empty - will be populated by reader stage
            )
            tasks.append(file_task)
        
        logger.info(f"Created {len(tasks)} file groups from {len(files)} files")
        return tasks
    
    def _get_file_list(self) -> list[str]:
        """Get the list of files to process."""
        if isinstance(self.file_paths, str):
            path = Path(self.file_paths)
            if path.is_file():
                return [str(path)]
            else:
                # Directory or pattern
                return get_all_files_paths_under(
                    self.file_paths,
                    recurse_subdirectories=True,
                    keep_extensions=self.file_extensions,
                    storage_options=self.storage_options,
                )
        else:
            # List of files
            return self.file_paths
    
    def _get_dataset_name(self, files: list[str]) -> str:
        """Extract dataset name from file paths."""
        if files:
            # Use the parent directory name or first file stem
            first_file = Path(files[0])
            if first_file.parent.name and first_file.parent.name != ".":
                return first_file.parent.name
            else:
                return first_file.stem
        return "dataset"
    
    def _partition_by_count(self, files: list[str], count: int) -> list[list[str]]:
        """Partition files by count."""
        partitions = []
        for i in range(0, len(files), count):
            partitions.append(files[i : i + count])
        return partitions
    
    def _partition_by_size(self, files: list[str], blocksize: int | str) -> list[list[str]]:
        """Partition files by target size.
        
        Note: This is a simplified implementation. A full implementation
        would check actual file sizes and create balanced partitions.
        """
        # Convert blocksize to bytes if string
        if isinstance(blocksize, str):
            blocksize = self._parse_size(blocksize)
        
        # For now, use a simple heuristic
        # Assume average JSONL file is ~100MB
        avg_file_size = 100 * 1024 * 1024  # 100MB
        files_per_block = max(1, blocksize // avg_file_size)
        
        return self._partition_by_count(files, files_per_block)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '128MB' to bytes."""
        size_str = size_str.upper().strip()
        
        units = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 * 1024,
            "GB": 1024 * 1024 * 1024,
            "TB": 1024 * 1024 * 1024 * 1024
        }
        
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                number = float(size_str[: -len(unit)])
                return int(number * multiplier)
        
        # If no unit, assume bytes
        return int(size_str)


@dataclass
class JsonlReaderStage(ProcessingStage[FileGroupTask]):
    """Stage that processes a group of JSONL files into a DocumentBatch.

    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.
    """
    
    text_column: str = "content"
    id_column: str | None = "id"
    additional_columns: list[str] = field(default_factory=list)
    columns: list[str] | None = None  # If specified, only read these columns
    reader: str = "pandas"  # "pandas" or "pyarrow"
    reader_kwargs: dict[str, Any] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        return "jsonl_reader"
    
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
        
        # Get storage options from task metadata
        storage_options = task.metadata.get("storage_options", {})

        # Read the files
        if self.reader == "pandas":
            df = self._read_with_pandas(task.file_paths, storage_options, self.reader_kwargs, self.columns)
        elif self.reader == "pyarrow":
            df = self._read_with_pyarrow(task.file_paths, self.reader_kwargs, self.columns)
        else:
            raise ValueError(f"Unknown reader: {self.reader}")
        
        if df is None or (hasattr(df, "empty") and df.empty) or (hasattr(df, "num_rows") and df.num_rows == 0):
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
            text_column=self.text_column,
            id_column=self.id_column,
            additional_columns=self.additional_columns,
        )
        batch.add_stage(self.name)
        
        return batch
    
    def _read_with_pandas(
        self, file_paths: list[str], storage_options: dict[str, Any], reader_kwargs: dict[str, Any], columns: list[str] | None
    ) -> pd.DataFrame | None:
        """Read JSONL files using pandas."""
        import pandas as pd
        
        dfs = []
        
        for file_path in file_paths:
            try:
                # Read the JSONL file
                df = pd.read_json(file_path, lines=True, storage_options=storage_options, **reader_kwargs)
                
                # Select only the specified columns if provided
                if columns is not None:
                    # Check which columns actually exist in the dataframe
                    existing_columns = [col for col in columns if col in df.columns]
                    missing_columns = [col for col in columns if col not in df.columns]
                    
                    if missing_columns:
                        logger.warning(f"Columns {missing_columns} not found in {file_path}")
                    
                    if existing_columns:
                        df = df[existing_columns]
                    else:
                        logger.error(f"None of the requested columns found in {file_path}")
                        continue
                
                dfs.append(df)
                logger.debug(f"Read {len(df)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue
        
        if not dfs:
            return None
        
        # Concatenate all dataframes
        return pd.concat(dfs, ignore_index=True)
    
    def _read_with_pyarrow(self, file_paths: list[str], reader_kwargs: dict[str, Any], columns: list[str] | None) -> pa.Table | None:
        """Read JSONL files using pyarrow."""
        import pyarrow as pa
        import pyarrow.json as pj
        
        tables = []
        
        for file_path in file_paths:
            try:
                # PyArrow JSON reader doesn't support column selection during read,
                # so we read all and then select
                table = pj.read_json(file_path, **reader_kwargs)
                
                # Select only the specified columns if provided
                if columns is not None:
                    # Check which columns actually exist in the table
                    existing_columns = [col for col in columns if col in table.column_names]
                    missing_columns = [col for col in columns if col not in table.column_names]
                    
                    if missing_columns:
                        logger.warning(f"Columns {missing_columns} not found in {file_path}")
                    
                    if existing_columns:
                        table = table.select(existing_columns)
                    else:
                        logger.error(f"None of the requested columns found in {file_path}")
                        continue
                
                tables.append(table)
                logger.debug(f"Read {len(table)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue
        
        if not tables:
            return None
        
        # Concatenate all tables
        return pa.concat_tables(tables)


@dataclass
class JsonlReader(CompositeStage[Task]):
    """Composite stage for reading JSONL files.
    
    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. JsonlReaderStage - reads file groups into DocumentBatches
    """
    
    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    text_column: str = "content"
    id_column: str | None = "id"
    additional_columns: list[str] | None = None
    columns: list[str] | None = None  # If specified, only read these columns
    reader: str = "pandas"  # "pandas" or "pyarrow"
    reader_kwargs: dict[str, Any] | None = None
    storage_options: dict[str, Any] | None = None
    
    @property
    def name(self) -> str:
        return "jsonl_reader"
    
    def decompose(self) -> list[ProcessingStage]:
        """Decompose into file partitioning and processing stages."""
        return [
            # First stage: partition files into groups
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=[".jsonl", ".json"],
                storage_options=self.storage_options,
            ),
            # Second stage: process file groups into document batches
            JsonlReaderStage(
                text_column=self.text_column,
                id_column=self.id_column,
                additional_columns=self.additional_columns or [],
                columns=self.columns,
                reader=self.reader,
                reader_kwargs=self.reader_kwargs or {},
            ),
        ]
    
    def get_description(self) -> str:
        """Get a description of this composite stage."""
        parts = [f"Read JSONL files from {self.file_paths}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        if self.columns:
            parts.append(f"reading columns: {self.columns}")
        return ", ".join(parts)
