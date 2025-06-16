from .dataframe import DataFrameWriter


class JsonlWriter(DataFrameWriter):
    """Writer that writes a DocumentBatch to a JSONL file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, format="jsonl", **kwargs)

    @property
    def name(self) -> str:
        return "jsonl_writer"
