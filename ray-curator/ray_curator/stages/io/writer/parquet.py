from .dataframe import DataFrameWriter


class ParquetWriter(DataFrameWriter):
    """Writer that writes a DocumentBatch to a Parquet file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, format="parquet", **kwargs)

    @property
    def name(self) -> str:
        return "parquet_writer"
