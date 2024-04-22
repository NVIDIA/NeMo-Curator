import dask.dataframe as dd
import pandas as pd
from dask.dataframe.utils import assert_eq

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


def all_equal(left_dataset, right_dataset):
    return all(left_dataset.df.compute() == right_dataset.df.compute())


class TestShuffling:
    def test_shuffle(self):
        original_dataset = list_to_dataset(["one", "two", "three", "four", "five"])
        expected_dataset = list_to_dataset(["two", "three", "one", "four", "five"])
        shuffle = nc.Shuffle(seed=42)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_new_partitions(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=3
        )
        expected_dataset = list_to_dataset(
            ["two", "three", "one", "four", "five"], npartitions=3
        )
        shuffle = nc.Shuffle(seed=42, npartitions=2)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_filename(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=1
        )
        original_dataset.df["filename"] = "original.jsonl"

        expected_data = {
            "text": ["one", "two", "three", "four", "five"],
            "filename": [
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000002.jsonl",
                "file_0000000002.jsonl",
            ],
        }
        pdf = pd.DataFrame(expected_data)
        expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

        shuffle = nc.Shuffle(seed=42, npartitions=2)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_custom_filenames(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=1
        )
        original_dataset.df["filename"] = "original.jsonl"

        expected_data = {
            "text": ["one", "two", "three", "four", "five"],
            "filename": [
                "my_1.test",
                "my_1.test",
                "my_1.test",
                "my_2.test",
                "my_2.test",
            ],
        }
        pdf = pd.DataFrame(expected_data)
        expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

        def filename_fn(x):
            return f"my_{x}.test"

        shuffle = nc.Shuffle(seed=42, npartitions=2, partition_to_filename=filename_fn)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)
