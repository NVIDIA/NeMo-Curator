import dask.dataframe as dd
import pandas as pd
from dask.dataframe.utils import assert_eq

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


class TestBlending:
    def test_blend_as_original():
        first_dataset = list_to_dataset(["one", "two", "three"])
        result_dataset = nc.blend_datasets(len(first_dataset), [first_dataset], [1.0])
        assert_eq(first_dataset, result_dataset)

    def test_equal_blend():
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            2, [first_dataset, second_dataset], [0.5, 0.5]
        )
        counts = result_dataset.df["text"].value_counts()
        assert len(result_dataset) == 2
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_equal_blend_with_weights():
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            2, [first_dataset, second_dataset], [2.0, 2.0]
        )
        counts = result_dataset.df["text"].value_counts()
        assert len(result_dataset) == 2
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_uneven_blend():
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            4, [first_dataset, second_dataset], [3.0, 1.0]
        )
        counts = result_dataset.df["text"].value_counts()
        assert len(result_dataset) == 4
        assert counts["a"] == 3
        assert counts["b"] == 1

    def test_very_uneven_blend():
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            4, [first_dataset, second_dataset], [1.0, 0.0]
        )
        counts = result_dataset.df["text"].value_counts()
        assert len(result_dataset) == 4
        assert counts["a"] == 4
        assert counts["b"] == 0
