from typing import Any

import pandas as pd
import pyarrow as pa


def _convert_numpy_to_native(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert numpy arrays and types to Python native types.

    This is needed because Ray Data converts Python lists to numpy arrays,
    which can cause issues with serialization and validation.

    Args:
        obj: Object that may contain numpy arrays/types

    Returns:
        Object with numpy arrays/types converted to Python native types
    """
    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Handle numpy scalar types
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass

    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {key: _convert_numpy_to_native(value) for key, value in obj.items()}

    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        converted = [_convert_numpy_to_native(item) for item in obj]
        return converted if isinstance(obj, list) else tuple(converted)

    # Handle other iterables (but not strings, bytes, pandas DataFrames, or PyArrow Tables)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, pd.DataFrame, pa.Table)):
        try:
            return [_convert_numpy_to_native(item) for item in obj]
        except (TypeError, AttributeError):
            pass

    # Return as-is if no conversion needed
    return obj
