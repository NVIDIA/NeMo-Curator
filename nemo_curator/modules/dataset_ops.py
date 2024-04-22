import math
from typing import Any, Callable, List, Optional

import dask.array as da
import dask.dataframe as dd

from nemo_curator.datasets.doc_dataset import DocumentDataset


def default_filename(partition_num: int) -> str:
    return f"file_{partition_num:010d}.jsonl"


class Shuffle:
    def __init__(
        self,
        seed: Optional[int] = None,
        npartitions: Optional[int] = None,
        partition_to_filename: Callable[[int], str] = default_filename,
    ) -> None:
        """
        Randomly permutes the dataset. This will make the original "filename" column invalid, so if the column is present it will be overwritten.
        Args:
            seed: The random seed that will be used to determine which file each datapoint goes to.
            npartitions: The output number of partitions to create in the dataset.
                If None, it will retain the same number of partitions as the original dataset.
            partition_to_filename: If the filename column is present, it will be overwritten.
                Passing a function in through this argument allows the user to configure what the filename
                will look like given the partition number. The default method names the partition
                f'file_{partition_num:010d}.jsonl' and should be changed if the user is not using a .jsonl format.
        """
        self.seed = seed
        self.npartitions = npartitions
        self.partition_to_filename = partition_to_filename

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        rand_col = "_shuffle_rand"
        new_npartitions = (
            dataset.df.npartitions if self.npartitions is None else self.npartitions
        )

        rng = da.random.default_rng(seed=self.seed)
        rand_array = rng.integers(0, new_npartitions, size=len(dataset.df))
        rand_df = dd.from_dask_array(rand_array, columns=[rand_col]).repartition(
            npartitions=dataset.df.npartitions
        )
        dataset.df[rand_col] = rand_df[rand_col]

        shuffled_df = dataset.df.shuffle(rand_col, npartitions=new_npartitions)
        shuffled_df.drop(columns=[rand_col])

        if "filename" in shuffled_df.columns:
            shuffled_df = shuffled_df.map_partitions(self._overwrite_filename)

        return DocumentDataset(shuffled_df)

    def _overwrite_filename(self, partition, partition_info=None):
        if partition_info is None:
            return partition

        filename = self.partition_to_filename(partition_info["number"])
        partition["filename"] = filename

        return partition


def blend_datasets(
    target_size: int, datasets: List[DocumentDataset], sampling_weights: List[float]
) -> DocumentDataset:
    """
    Combined multiple datasets into one with different amounts of each dataset
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
        raise ValueError(
            f"Different number of datasets and weights specified. {len(datasets)} datasets and {len(sampling_weights)}"
        )

    weight_sum = sum(sampling_weights)
    sampling_weights = [weight / weight_sum for weight in sampling_weights]
    num_documents_per_dataset = [
        math.ceil(weight * target_size) for weight in sampling_weights
    ]

    blend_components = []
    for dataset, num_documents in zip(datasets, num_documents_per_dataset):
        # Repeatedly sample from the dataset
        num_epochs = math.ceil(num_documents / len(dataset))
        for _ in range(num_epochs):
            sample = dataset.df.head(n=num_documents, npartitions=-1, compute=False)
            blend_components.append(sample)
            num_documents -= len(sample)

    blended_dataset = dd.concat(blend_components)

    return DocumentDataset(blended_dataset)
