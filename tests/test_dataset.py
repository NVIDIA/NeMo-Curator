import dask.dataframe as dd
import pandas as pd
import pytest

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")


def all_equal(left_result: pd.DataFrame, right_result: pd.DataFrame, gpu=True):
    l_cols = set(left_result.columns)
    r_cols = set(right_result.columns)
    assert l_cols == r_cols

    for col in left_result.columns:
        left = left_result[col].reset_index(drop=True)
        right = right_result[col].reset_index(drop=True)

        # The `all` function expects an iterable, so we need to convert cuDF to Pandas
        if gpu:
            left = left.to_pandas()
            right = right.to_pandas()

        assert all(left == right), f"Mismatch in {col} column.\n{left}\n{right}\n"


class TestDocumentDataset:
    def test_to_from_pandas(self):
        original_df = pd.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        dataset = DocumentDataset.from_pandas(original_df)
        converted_df = dataset.to_pandas()
        pd.testing.assert_frame_equal(original_df, converted_df)

    def test_init_pandas(self):
        original_df = pd.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        with pytest.raises(RuntimeError):
            dataset = DocumentDataset(dataset_df=original_df)

    def test_init_dask(self):
        original_df = pd.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        ddf = dd.from_pandas(original_df, npartitions=1)
        dataset = DocumentDataset(dataset_df=ddf)
        assert type(dataset.df == dd.DataFrame)
        pd.testing.assert_frame_equal(original_df, dataset.df.compute())

    @pytest.mark.gpu
    def test_to_from_cudf(self):
        original_df = cudf.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        dataset = DocumentDataset.from_cudf(original_df)
        converted_df = dataset.to_cudf()
        all_equal(original_df, converted_df, gpu=True)

    @pytest.mark.gpu
    def test_init_cudf(self):
        original_df = cudf.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        with pytest.raises(RuntimeError):
            dataset = DocumentDataset(dataset_df=original_df)

    @pytest.mark.gpu
    def test_init_dask_cudf(self):
        original_df = cudf.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        ddf = dask_cudf.from_cudf(original_df, npartitions=1)
        dataset = DocumentDataset(dataset_df=ddf)
        assert type(dataset.df == dask_cudf.DataFrame)
        all_equal(original_df, dataset.df.compute(), gpu=True)
