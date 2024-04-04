import math
from typing import List

import dask.dataframe as dd

from nemo_curator.datasets.doc_dataset import DocumentDataset


def blend_datasets(
    target_size: int, datasets: List[DocumentDataset], sampling_weights: List[float]
) -> DocumentDataset:
    """
    Combined multiple datasets into one with different amounts of each dataset
    Args:
        target_size: The number of documents the resulting dataset should have
        datasets: A list of all datasets to combine together
        sampling_weights: A list of weights to assign to each dataset in the input. Weights will be
            normalized across the whole list as a part of the sampling process. For example, if the normalized
            sampling weight for dataset 1 is 0.02, 2% ofthe total samples will be sampled from dataset 1.
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

    # TODO: Shuffle the dataset

    return DocumentDataset(blended_dataset)
