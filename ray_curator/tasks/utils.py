""" This file contains utility functions for tasks. Majorly for converting between different task types. """

import pandas as pd
import pyarrow as pa
from typing import Any


def get_columns(data: Any) -> list[str]:
    """Get column names from the data."""
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    elif isinstance(data, pa.Table):
        return data.column_names
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
