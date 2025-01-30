import warnings
from abc import ABC
from typing import Optional

import dask.dataframe as dd

from nemo_curator.datasets.doc_dataset import DocumentDataset


def _perform_removal(
    left: dd.DataFrame,
    duplicates: dd.DataFrame,
    id_field: str,
    group_field: str,
) -> dd.DataFrame:
    # Create a new column name for temporary ID storage during merge
    new_id_field = f"{id_field}_new"

    duplicates_to_remove = (
        duplicates
        # Redistribute data across partitions so that all duplicates are in same partition
        .shuffle(on=[group_field], ignore_index=True)
        # For each partition, keep only the duplicated rows (excluding first occurrence)
        .map_partitions(lambda x: x[x[group_field].duplicated(keep="first")]).drop(
            columns=group_field
        )
        # Rename the ID field to avoid conflicts in the upcoming merge
        .rename(columns={id_field: new_id_field})[[new_id_field]]
    )

    merge = left.merge(
        right=duplicates_to_remove,
        how="left",
        broadcast=True,  # Broadcast smaller DataFrame to all partitions
        left_on=id_field,
        right_on=new_id_field,
    )

    # This effectively removes all rows that were not in duplicates_to_remove
    removed_result = merge[merge[new_id_field].isna()].drop(columns=[new_id_field])
    return removed_result


class Deduplicator(ABC):
    def __init__(
        self,
        id_field: str,
        text_field: str,
        grouped_field: str,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.id_field = id_field
        self.text_field = text_field
        self.grouped_field = grouped_field
        self.cache_dir = cache_dir

    def identify(self, *args, **kwargs):
        """Abstract method to be implemented by concrete deduplicator classes.
        Should implement the logic for identifying duplicates in the dataset."""
        raise NotImplementedError

    def remove(
        self, dataset: DocumentDataset, duplicates: DocumentDataset
    ) -> DocumentDataset:
        """
        Remove duplicate documents from the dataset based on identified duplicate groups.

        Parameters
        ----------
        dataset: DocumentDataset
            The input datset to remove duplicates from.

        duplicates: DocumentDataset
            The dataset containing IDs of all documents and the corresponding duplicate group
            they belong to. Documents in the same group are considered duplicates.
            Only the first document in each group is retained.

        Returns
        -------
        DocumentDataset of all documents with duplicates removed.
        """
        if self.cache_dir is None:
            msg = "Cache directory should be specified for improved performance for removal step."
            warnings.warn(msg)

        left = dataset.df
        right = duplicates.df

        print(f"{left.npartitions=}, {right.npartitions=}")
        if left.npartitions < right.npartitions:
            msg = (
                "The number of partitions in the dataset to remove duplicates from is less than the number of partitions in the duplicates dataset. "
                "This may lead to a shuffle join. Please re-read the datasets and call nemo_curator._deduplicat.perform_merge explicitly."
            )
            raise ValueError(msg)

        removed_result = _perform_removal(
            left=left,
            duplicates=right,
            id_field=self.id_field,
            group_field=self.grouped_field,
        )
        return DocumentDataset(removed_result)

    def __call__(
        self, dataset: DocumentDataset, perform_removal: bool = False
    ) -> DocumentDataset:
        """
        Main entry point for deduplication process.

        Parameters
        ----------
        dataset: DocumentDataset
            The input datset to remove duplicates from.
        perform_removal: bool
            If True, duplicates are removed from the dataset.
            If False, only the duplicates are identified.

        Returns
        -------
        DocumentDataset of all duplicates (id field, group field) if perform_removal is False.
        DocumentDataset of all documents with duplicates removed if perform_removal is True.
        """
        # First identify duplicates
        duplicates = self.identify(dataset)
        # Then optionally remove them
        if perform_removal:
            return self.remove(dataset, duplicates)
        return duplicates
