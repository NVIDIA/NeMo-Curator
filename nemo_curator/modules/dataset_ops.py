# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable

import dask.dataframe as dd
import numpy as np
import pandas as pd

from nemo_curator.datasets.doc_dataset import DocumentDataset
from nemo_curator.modules.base import BaseModule


def default_filename(partition_num: int) -> str:
    return f"file_{partition_num:010d}.jsonl"


class Shuffle(BaseModule):
    def __init__(
        self,
        seed: int | None = None,
        npartitions: int | None = None,
        partition_to_filename: Callable[[int], str] = default_filename,
        filename_col: str = "file_name",
    ) -> None:
        """
        Randomly permutes the dataset. This will make the original filename_col column invalid, so if the column is present it will be overwritten.
        Args:
            seed: The random seed that will be used to determine which partition (file) each datapoint goes to.
                Setting the seed will guarantee determinism, but may be slightly slower (20-30% slower)
                depending on the dataset size.
            npartitions: The output number of partitions to create in the dataset.
                If None, it will retain the same number of partitions as the original dataset.
            partition_to_filename: If the filename column is present, it will be overwritten.
                Passing a function in through this argument allows the user to configure what the filename
                will look like given the partition number. The default method names the partition
                f'file_{partition_num:010d}.jsonl' and should be changed if the user is not using a .jsonl format.
        """
        super().__init__(input_backend="pandas")
        self.seed = seed
        self.npartitions = npartitions
        self.partition_to_filename = partition_to_filename
        self.rand_col = "_shuffle_rand"
        self.filename_col = filename_col

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        if self.seed is None:
            return self.shuffle_nondeterministic(dataset)
        else:
            return self.shuffle_deterministic(dataset)

    def shuffle_deterministic(self, dataset: DocumentDataset) -> DocumentDataset:
        new_npartitions = dataset.df.npartitions if self.npartitions is None else self.npartitions

        dataset.df[self.rand_col] = dataset.df.map_partitions(self._add_rand_col)

        shuffled_df = dataset.df.set_index(self.rand_col, npartitions=new_npartitions)
        shuffled_df = shuffled_df.reset_index(drop=True)

        if self.filename_col in shuffled_df:
            shuffled_df[self.filename_col] = shuffled_df.map_partitions(self._add_filename)

        return DocumentDataset(shuffled_df)

    def shuffle_nondeterministic(self, dataset: DocumentDataset) -> DocumentDataset:
        new_npartitions = dataset.df.npartitions if self.npartitions is None else self.npartitions

        dataset.df[self.rand_col] = dataset.df.map_partitions(self._add_rand_col)

        shuffled_df = dataset.df.shuffle(self.rand_col, npartitions=new_npartitions, ignore_index=True)
        shuffled_df = shuffled_df.drop(columns=[self.rand_col])
        shuffled_df = shuffled_df.map_partitions(self._partition_shuffle)

        return DocumentDataset(shuffled_df)

    def _add_rand_col(self, partition: pd.DataFrame, partition_info: dict | None = None) -> pd.Series:
        if partition_info is None:
            partition_info = {
                "number": 0,
            }

        if self.seed is not None:
            np.random.seed(self.seed + partition_info["number"])  # noqa: NPY002
        return np.random.randint(0, np.iinfo("int64").max, size=len(partition))  # noqa: NPY002

    def _partition_shuffle(self, partition: pd.DataFrame, partition_info: dict | None = None) -> pd.DataFrame:
        if partition_info is None:
            return partition

        partition_num = partition_info["number"]
        random_state = self.seed + partition_num if self.seed is not None else None

        partition = partition.sample(frac=1, random_state=random_state).reset_index(drop=True)

        if self.filename_col in partition:
            filename = self.partition_to_filename(partition_num)
            partition[self.filename_col] = filename

        return partition

    def _add_filename(self, partition: pd.DataFrame, partition_info: dict | None = None) -> list[str]:
        if partition_info is None:
            return [self.filename_col] * len(partition)

        filename = self.partition_to_filename(partition_info["number"])

        return [filename for _ in range(len(partition))]


def blend_datasets(
    target_size: int, datasets: list[DocumentDataset], sampling_weights: list[float]
) -> DocumentDataset:
    """
    Combines multiple datasets into one with different amounts of each dataset.
    Args:
        target_size: The number of documents the resulting dataset should have.
            The actual size of the dataset may be slightly larger if the normalized weights do not allow
            for even mixtures of the datasets.
        datasets: A list of all datasets to combine together
        sampling_weights: A list of weights to assign to each dataset in the input. Weights will be
            normalized across the whole list as a part of the sampling process. For example, if the normalized
            sampling weight for dataset 1 is 0.02, 2% ofthe total samples will be sampled from dataset 1.
            There are guaranteed to be math.ceil(normalized_weight_i * target_size) elements from dataset i in
            the final blend.
    """
    if len(datasets) != len(sampling_weights):
        msg = (
            f"Different number of datasets and weights specified. {len(datasets)} datasets and {len(sampling_weights)}"
        )
        raise ValueError(msg)

    weight_sum = sum(sampling_weights)
    sampling_weights = [weight / weight_sum for weight in sampling_weights]
    num_documents_per_dataset = [math.ceil(weight * target_size) for weight in sampling_weights]

    blend_components = []
    for dataset, num_documents in zip(datasets, num_documents_per_dataset, strict=False):
        remaining_documents = num_documents
        # Repeatedly sample from the dataset
        while remaining_documents > 0:
            sample = _partition_head(dataset.df, remaining_documents)
            blend_components.append(sample)
            remaining_documents -= len(sample)

    blended_dataset = dd.concat(blend_components)

    return DocumentDataset(blended_dataset)


def _partition_head(ddf: dd.DataFrame, n: int) -> dd.DataFrame:
    """
    Returns the first n rows in a dataframe while preserving the partitions.
    Meant as a replacement for ddf.head(npartitions=-1, compute=False) as it
    uses too much memory at large scales

    Args:
        ddf: The dataframe to get the first rows from
        n: The number of rows to get
    """
    original_meta = ddf.dtypes.to_dict()
    partition_lengths = ddf.map_partitions(len)
    num_partitions = 0
    total_size = 0
    last_length = 0
    for length in partition_lengths:
        total_size += length
        num_partitions += 1
        last_length = length
        if total_size >= n:
            break

    delayed_df = ddf.to_delayed()
    excess_elems = max(0, total_size - n)
    delayed_df = delayed_df[:num_partitions]
    delayed_df[-1] = delayed_df[-1].head(last_length - excess_elems)

    return dd.from_delayed(delayed_df, meta=original_meta)
