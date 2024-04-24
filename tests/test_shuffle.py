import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset

# Single threaded Dask is the only way to guarantee shuffle determinism
# Docs: https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.shuffle.html
client = Client(n_workers=1, threads_per_worker=1)


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


def all_equal(left_dataset, right_dataset):
    left_result = left_dataset.df.compute()
    right_result = right_dataset.df.compute()

    l_cols = set(left_result.columns)
    r_cols = set(right_result.columns)
    assert l_cols == r_cols
    for col in left_result.columns:
        left = left_result[col].reset_index(drop=True)
        right = right_result[col].reset_index(drop=True)
        assert all(left == right), f"Mismatch in {col} column.\n{left}\n{right}\n"


class TestShuffling:
    def test_shuffle(self):
        original_dataset = list_to_dataset(["one", "two", "three", "four", "five"])
        expected_dataset = list_to_dataset(["four", "three", "two", "one", "five"])
        shuffle = nc.Shuffle(seed=42)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_new_partitions(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=3
        )
        expected_dataset = list_to_dataset(
            ["four", "three", "two", "one", "five"], npartitions=3
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
            "text": ["four", "three", "two", "one", "five"],
            "filename": [
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
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
            "text": ["four", "three", "two", "one", "five"],
            "filename": [
                "my_1.test",
                "my_1.test",
                "my_1.test",
                "my_1.test",
                "my_1.test",
            ],
        }
        pdf = pd.DataFrame(expected_data)
        expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

        def filename_fn(x):
            return f"my_{x}.test"

        shuffle = nc.Shuffle(seed=42, npartitions=2, partition_to_filename=filename_fn)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)
