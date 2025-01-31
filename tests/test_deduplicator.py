import random

import pandas as pd
import pytest
from dask import dataframe as dd

from nemo_curator.modules.removal import remove_duplicates


@pytest.fixture()
def ids():
    # Dataset has id a0...a9, b0...b9, c0...c9, d0...d9
    l = [f"{group}{i}" for group in ["a", "b", "c", "d"] for i in range(10)]
    # We shuffle it to make sure all duplicates are not in the same partition
    random.shuffle(l)
    return l


@pytest.fixture
def sample_data(ids):
    df = pd.DataFrame(
        {
            "id": ids,
            "text": [f"text for {_id}" for _id in ids],
        }
    )
    return dd.from_pandas(df, npartitions=4)


@pytest.fixture
def duplicate_data(ids):
    # In each group we want to keep only the first occurrence (e.g. a1, b1, c1, d1)
    df = pd.DataFrame([{"id": _id, "group": _id[0]} for _id in ids])
    # Shuffle to make sure all duplicates are not in the same partition
    return dd.from_pandas(df, npartitions=2)


def test_remove_duplicates_basic(
    sample_data: dd.DataFrame, duplicate_data: dd.DataFrame
):
    # Test basic duplicate removal functionality
    result = remove_duplicates(
        left=sample_data, duplicates=duplicate_data, id_field="id", group_field="group"
    )

    result = result.compute()

    assert list(result.columns) == ["id", "text"]
    assert len(result) == 4
    # It's not guaranteed that we'll have a0, b0, c0, d0 in the result
    # So we should check the first character
    assert set(result["id"].apply(lambda x: x[0]).tolist()) == set(["a", "b", "c", "d"])


def test_remove_duplicates_all_duplicates(ids: list[str], sample_data: dd.DataFrame):
    duplicates = dd.from_pandas(
        pd.DataFrame({"id": ids, "group": [1] * len(ids)}), npartitions=2
    )

    result = remove_duplicates(
        left=sample_data, duplicates=duplicates, id_field="id", group_field="group"
    )

    result = result.compute()
    assert list(result.columns) == ["id", "text"]
    # Should keep only one of the occurrences
    assert len(result) == 1


def test_not_remove_duplicates_unique(ids: list[str], sample_data: dd.DataFrame):
    # We create a dataset where first 30 ids are in one group
    # Next 9 ids are in distinct groups
    # And last id is not mentioned in duplicates

    duplicates = dd.from_pandas(
        pd.DataFrame(
            {
                "id": ids[:30] + ids[30:39],
                "group": ["group0"] * 30 + [f"group{i}" for i in range(1, 10)],
            }
        ),
        npartitions=2,
    )
    result = remove_duplicates(
        left=sample_data, duplicates=duplicates, id_field="id", group_field="group"
    )

    result = result.compute()
    assert list(result.columns) == ["id", "text"]
    # It has 1 row from the first group of 30
    # 9 rows from the 9 distinct groups
    # And 1 row from the last group which is not included in set of duplicates
    assert len(result) == 1 + 9 + 1
    # The last 10 ids should be in the result, there would be one more from the first 30
    assert set(ids[30:]).issubset(set(result["id"].tolist()))


def test_remove_duplicates_raise_error():
    # Create sample dataframes with specific partition counts
    df1 = dd.from_pandas(
        pd.DataFrame({"id": ["a1", "a2", "a3"], "text": ["text1", "text2", "text3"]}),
        npartitions=2,
    )  # dataset with 2 partitions

    duplicates = dd.from_pandas(
        pd.DataFrame(
            {"id": ["a1", "a2", "a3"], "group": ["group1", "group1", "group1"]}
        ),
        npartitions=3,
    )  # duplicates dataset with 3 partitions

    # Test that it raises ValueError when right npartitions are greater than left npartitions
    with pytest.raises(ValueError) as exc_info:
        remove_duplicates(
            left=df1,
            duplicates=duplicates,
            id_field="id",
            group_field="group",
        )

    expected_msg = (
        "The number of partitions in `left` is less than the number of partitions in the duplicates dataset. "
        "This may lead to a shuffle join. Please re-read left and right with different partition sizes, or repartition left / right."
    )
    assert str(exc_info.value) == expected_msg
