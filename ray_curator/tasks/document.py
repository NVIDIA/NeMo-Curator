from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
from loguru import logger

from .tasks import Task


@dataclass
class DocumentBatch(Task[pa.Table | pd.DataFrame]):
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
    id_column: str | None = None
    additional_columns: list[str] = field(default_factory=list)
    data: pa.Table | pd.DataFrame = field(default_factory=pa.Table)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pyarrow(self) -> pa.Table:
        """Convert data to PyArrow table."""
        if isinstance(self.data, pa.Table):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return pa.Table.from_pandas(self.data)
        else:
            msg = f"Cannot convert {type(self.data)} to PyArrow table"
            raise TypeError(msg)

    def to_pandas(self) -> pd.DataFrame:
        """Convert data to Pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data
        elif isinstance(self.data, pa.Table):
            return self.data.to_pandas()
        else:
            msg = f"Cannot convert {type(self.data)} to Pandas DataFrame"
            raise TypeError(msg)

    @property
    def num_items(self) -> int:
        """Get the number of documents in this batch."""
        return len(self.data)

    def get_columns(self) -> list[str]:
        """Get column names from the data."""
        if isinstance(self.data, pd.DataFrame):
            return list(self.data.columns)
        elif isinstance(self.data, pa.Table):
            return self.data.column_names
        else:
            msg = f"Unsupported data type: {type(self.data)}"
            raise TypeError(msg)

    def validate(self) -> bool:
        """Validate the task data."""
        missing_columns = []
        cols_to_check = [*self.additional_columns, self.text_column]
        if self.id_column:
            cols_to_check.append(self.id_column)
        for col in cols_to_check:
            if col not in self.get_columns():
                missing_columns.append(col)
        if missing_columns:
            logger.warning(f"Task {self.task_id} missing required columns: {missing_columns}")
            return False
        return True
