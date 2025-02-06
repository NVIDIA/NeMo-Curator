import dask.dataframe as dd


def remove_duplicates(
    left: dd.DataFrame,
    duplicates: dd.DataFrame,
    id_field: str,
    group_field: str,
) -> dd.DataFrame:
    if left.npartitions < duplicates.npartitions:
        msg = (
            "The number of partitions in `left` is less than the number of partitions in the duplicates dataset. "
            "This may lead to a shuffle join. Please re-read left and right with different partition sizes, or repartition left / right."
        )
        raise ValueError(msg)

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
