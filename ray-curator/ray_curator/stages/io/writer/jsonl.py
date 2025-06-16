from dataclasses import dataclass, field
from typing import Any

from ray_curator.tasks import DocumentBatch

from .dataframe import BaseWriter


@dataclass
class JsonlWriter(BaseWriter):
    """Writer that writes a DocumentBatch to a JSONL file."""

    # Additional kwargs for pandas.DataFrame.to_json
    file_extension: str = "jsonl"
    jsonl_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "jsonl_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write data to JSONL file using pandas DataFrame.to_json."""
        df = task.to_pandas()  # Convert to pandas DataFrame if needed

        # Build kwargs for to_json with explicit options
        json_kwargs = {
            "lines": True,
            "orient": "records",
            "storage_options": self.storage_options,
        }

        # Add any additional kwargs, allowing them to override defaults
        json_kwargs.update(self.jsonl_kwargs)

        df.to_json(file_path, **json_kwargs)
