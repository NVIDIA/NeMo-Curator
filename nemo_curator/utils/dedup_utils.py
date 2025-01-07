import uuid
from typing import List, Literal, Optional

import dask.dataframe as dd
from dask.distributed import wait

from nemo_curator.datasets.doc_dataset import DocumentDataset


def remove_duplicates(
    ds: DocumentDataset,
    duplicates: dd.DataFrame,
    join_key: str,
    deduplicate_column_name: Optional[str] = None,
):
    """
    Remove duplicates from a dataset based on the output of the deduplication module.
    Args:
        ds: The original dataset which has the `join_key`
        duplicates: This can be the output of the deduplication module, which contains the `join_key` and a `deduplicate_column_name` indicating which documents are grouped together.
            Or it can be set of duplicates (only `join_key`) that we wish to remove.
        join_key: The column name that was used to join the original dataset with the duplicates
        deduplicate_column_name: The column name that was used to deduplicate the duplicates.
            If None, it is assumed that the duplicates have been removed already.
    Returns:
        The original dataset with the duplicates removed
    """

    if deduplicate_column_name is not None:
        # Deduplicate the duplicates (output of exact / fuzzy dedup)
        duplicates_to_remove = duplicates.map_partitions(
            lambda x: x[x[deduplicate_column_name].duplicated(keep="first")]
        ).drop(columns=[deduplicate_column_name])
    else:
        duplicates_to_remove = duplicates

    # Rename the column join_key to avoid conflicts
    new_join_key = f"{join_key}_{uuid.uuid4().hex[:4]}"
    # Repartition to 1 to avoid memory issues
    duplicates_renamed = (
        duplicates_to_remove.rename(columns={join_key: new_join_key})
        .repartition(npartitions=1)
        .persist()
    )

    wait(duplicates_renamed)

    # Merge the deduplicated duplicates with the original dataset
    results = ds.merge(
        duplicates_renamed,
        how="left",
        left_on=join_key,
        right_on=new_join_key,
    )
    removed_result = results[results[new_join_key].isna()].drop(columns=[new_join_key])

    return removed_result


def remove_duplicates_scratch(
    files: List[str],
    input_file_type: Literal["parquet", "jsonl"],
    duplicates: dd.DataFrame,
    join_key: str,
    deduplicate_column_name: Optional[str] = None,
):
    """
    Remove duplicates from a dataset based on the output of the deduplication module.
    Args:
        files: The list of files that contain the original dataset which has the `join_key`
        input_file_type: The file type of the input files
        duplicates: This can be the output of the deduplication module, which contains the `join_key` and a `deduplicate_column_name` indicating which documents are grouped together.
            Or it can be set of duplicates (only `join_key`) that we wish to remove.
        join_key: The column name that was used to join the original dataset with the duplicates
        deduplicate_column_name: The column name that was used to deduplicate the duplicates.
            If None, it is assumed that the duplicates have been removed already.
    Returns:
        The original dataset with the duplicates removed
    """

    if input_file_type == "jsonl":
        read_fn = DocumentDataset.read_json
    elif input_file_type == "parquet":
        read_fn = DocumentDataset.read_parquet
    else:
        msg = f"Input file type {input_file_type} is not supported"
        raise ValueError(msg)

    # Read the original dataset
    ds = read_fn(files, blocksize="512MiB")

    # Remove duplicates
    return remove_duplicates(ds, duplicates, join_key, deduplicate_column_name)
