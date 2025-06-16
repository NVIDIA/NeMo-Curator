from dataclasses import dataclass, field
from typing import Any

from ray_curator.tasks import DocumentBatch

from .dataframe import BaseWriter


@dataclass
class ParquetWriter(BaseWriter):
    """Writer that writes a DocumentBatch to a Parquet file using pandas."""

    # Additional kwargs for pandas.DataFrame.to_parquet
    parquet_kwargs: dict[str, Any] = field(default_factory=dict)
    file_extension: str = "parquet"

    @property
    def name(self) -> str:
        return "parquet_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to Parquet file using pandas DataFrame.to_parquet."""
        df = task.to_pandas()  # Convert to pandas DataFrame if needed

        # Build kwargs for to_parquet with explicit options
        write_kwargs = {
            "index": None,
            "storage_options": self.storage_options,
        }

        # Add any additional kwargs, allowing them to override defaults
        write_kwargs.update(self.parquet_kwargs)

        df.to_parquet(file_path, **write_kwargs)
