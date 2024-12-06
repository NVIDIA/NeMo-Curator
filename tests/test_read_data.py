import random
import tempfile

import pandas as pd
import pytest

from nemo_curator.utils.distributed_utils import (
    read_data,
    read_data_blocksize,
    read_data_fpp,
)
from nemo_curator.utils.file_utils import get_all_files_paths_under

NUM_FILES = 5
NUM_RECORDS = 100


# Fixture to create multiple small JSONL files
@pytest.fixture
def mock_multiple_jsonl_files(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        jsonl_file = tmp_path / f"test_{file_id}.jsonl"
        with open(jsonl_file, "w") as f:
            for record_id in range(NUM_RECORDS):
                # 100 rows are ~5kb
                f.write(
                    f'{{"id": "id_{file_id}_{record_id}", "text": "A longish string {file_id}_{record_id}"}}\n'
                )
        file_paths.append(str(jsonl_file))
    return file_paths


# Fixture to create multiple small Parquet files
@pytest.fixture
def mock_multiple_parquet_files(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        # 100 rows are ~5kb
        parquet_file = tmp_path / f"test_{file_id}.parquet"
        df = pd.DataFrame(
            [
                {
                    "id": f"id_{file_id}_{record_id}",
                    "text": f"A string {file_id}_{record_id}",
                }
                for record_id in range(NUM_RECORDS)
            ]
        )
        # We specify row_group_size so that we can test splitting a single big file into smaller chunks
        df.to_parquet(parquet_file, compression=None, row_group_size=10)
        file_paths.append(str(parquet_file))
    return file_paths


@pytest.fixture
def mock_multiple_jsonl_files_different_cols(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        jsonl_file = tmp_path / f"test_diff_cols_{file_id}.jsonl"
        with open(jsonl_file, "w") as f:
            for record_id in range(NUM_RECORDS):
                # 100 rows are ~5kb
                f.write(
                    f'{{"col_{file_id}" : "some_col", "id": "id_{file_id}_{record_id}", "text": "A longish string {file_id}_{record_id}"}}\n'
                )
        file_paths.append(str(jsonl_file))
    return file_paths


# Fixture to create multiple small Parquet files
@pytest.fixture
def mock_multiple_parquet_files_different_cols(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        # 100 rows are ~5kb
        parquet_file = tmp_path / f"test_diff_cols_{file_id}.parquet"
        df = pd.DataFrame(
            [
                {
                    **(
                        {f"col_{file_id}": "some_col"}
                        if file_id != 0
                        else {"meta": "meta_col"}
                    ),
                    "id": f"id_{file_id}_{record_id}",
                    "text": f"A string {file_id}_{record_id}",
                }
                for record_id in range(NUM_RECORDS)
            ]
        )
        # We specify row_group_size so that we can test splitting a single big file into smaller chunks
        df.to_parquet(parquet_file, compression=None, row_group_size=10)
        file_paths.append(str(parquet_file))
    return file_paths


@pytest.mark.gpu
@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("blocksize", ["1kb", "5kb", "10kb"])
def test_cudf_read_data_blocksize_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, file_type, blocksize
):
    import cudf

    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_blocksize(
        input_files=input_files,
        backend="cudf",
        file_type=file_type,
        blocksize=blocksize,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.optimize().npartitions
    # Assert that we have two partitions (since we have ~15KB total data and a blocksize of 10KB)
    if blocksize == "1kb":
        assert (
            num_partitions > NUM_FILES
        ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "5kb":
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "10kb":
        assert (
            num_partitions < NUM_FILES
        ), f"Expected < {NUM_FILES} partitions but got {num_partitions}"
    else:
        raise ValueError(f"Invalid blocksize: {blocksize}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"

    assert isinstance(df["id"].compute(), cudf.Series)


@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("blocksize", ["1kb", "5kb", "10kb"])
def test_pandas_read_data_blocksize_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, file_type, blocksize
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_blocksize(
        input_files=input_files,
        backend="pandas",
        file_type=file_type,
        blocksize=blocksize,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.npartitions
    # Our total data is ~25kb where each file is 5kb
    if blocksize == "1kb":
        assert (
            num_partitions > NUM_FILES
        ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "5kb":
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "10kb":
        # Because pandas doesn't suppport reading json files together, a partition will only be as big as a single file
        if file_type == "jsonl":
            assert (
                num_partitions == NUM_FILES
            ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
        # Parquet files can be read together
        elif file_type == "parquet":
            assert (
                num_partitions < NUM_FILES
            ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    else:
        raise ValueError(f"Invalid blocksize: {blocksize}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"

    assert isinstance(df["id"].compute(), pd.Series)


@pytest.mark.parametrize(
    "backend",
    ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)],
)
@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("fpp", [1, NUM_FILES // 2, NUM_FILES, NUM_FILES * 2])
def test_read_data_fpp_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, backend, file_type, fpp
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_fpp(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        files_per_partition=fpp,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.npartitions
    # Assert that we have two partitions (since we have ~15KB total data and a blocksize of 10KB)
    if fpp == 1:
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif fpp == NUM_FILES // 2:
        assert (
            num_partitions < NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif fpp >= NUM_FILES:
        assert num_partitions == 1, f"Expected 1 partition but got {num_partitions}"
    else:
        raise ValueError(f"Invalid fpp: {fpp}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"
    if backend == "cudf":
        import cudf

        assert isinstance(df["id"].compute(), cudf.Series)
    elif backend == "pandas":
        assert isinstance(df["id"].compute(), pd.Series)


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
def test_read_data_blocksize_add_filename_jsonl(mock_multiple_jsonl_files, backend):
    df = read_data_blocksize(
        input_files=mock_multiple_jsonl_files,
        backend=backend,
        file_type="jsonl",
        blocksize="128Mib",
        add_filename=True,
        input_meta=None,
        columns=None,
    )

    assert "filename" in df.columns
    file_names = df["filename"].unique().compute()
    if backend == "cudf":
        file_names = file_names.to_pandas()

    assert len(file_names) == NUM_FILES
    assert set(file_names.values) == {
        f"test_{file_id}.jsonl" for file_id in range(NUM_FILES)
    }


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
def test_read_data_blocksize_add_filename_parquet(mock_multiple_parquet_files, backend):
    with pytest.raises(
        ValueError,
        match="add_filename and blocksize cannot be set at the same time for parquet files",
    ):
        read_data_blocksize(
            input_files=mock_multiple_parquet_files,
            backend=backend,
            file_type="parquet",
            blocksize="128Mib",
            add_filename=True,
            input_meta=None,
            columns=None,
        )


@pytest.mark.parametrize(
    "backend,file_type",
    [
        pytest.param("cudf", "jsonl", marks=pytest.mark.gpu),
        pytest.param("cudf", "parquet", marks=pytest.mark.gpu),
        ("pandas", "jsonl"),
        pytest.param(
            "pandas",
            "parquet",
            marks=pytest.mark.xfail(
                reason="filename column inaccessible with pandas backend and parquet"
            ),
        ),
    ],
)
def test_read_data_fpp_add_filename(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, backend, file_type
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_fpp(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        files_per_partition=NUM_FILES,
        add_filename=True,
        input_meta=None,
        columns=None,
    )

    assert set(df.head().columns) == {"filename", "id", "text"}
    file_names = df["filename"].unique().compute()
    if backend == "cudf":
        file_names = file_names.to_pandas()

    assert len(file_names) == NUM_FILES
    assert set(file_names.values) == {
        f"test_{file_id}.{file_type}" for file_id in range(NUM_FILES)
    }


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "file_type,add_filename,function_name",
    [
        *[("jsonl", True, func) for func in ["read_data_blocksize", "read_data_fpp"]],
        *[("jsonl", False, func) for func in ["read_data_blocksize", "read_data_fpp"]],
        *[
            ("parquet", False, func)
            for func in ["read_data_blocksize", "read_data_fpp"]
        ],
        *[("parquet", True, "read_data_fpp")],
    ],
)
@pytest.mark.parametrize(
    "cols_to_select", [None, ["id"], ["text", "id"], ["id", "text"]]
)
def test_read_data_select_columns(
    mock_multiple_jsonl_files,
    mock_multiple_parquet_files,
    backend,
    file_type,
    add_filename,
    function_name,
    cols_to_select,
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )
    if function_name == "read_data_fpp":
        func = read_data_fpp
        read_kwargs = {"files_per_partition": 1}
    elif function_name == "read_data_blocksize":
        func = read_data_blocksize
        read_kwargs = {"blocksize": "128Mib"}

    df = func(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        add_filename=add_filename,
        input_meta=None,
        columns=list(cols_to_select) if cols_to_select else None,
        **read_kwargs,
    )
    if not cols_to_select:
        cols_to_select = ["id", "text"]

    if not add_filename:
        # assert list(df.columns) == sorted(cols_to_select)
        assert list(df.head().columns) == sorted(cols_to_select)
    else:
        # assert list(df.columns) == sorted(cols_to_select + ["filename"])
        assert list(df.head().columns) == sorted(cols_to_select + ["filename"])


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("function_name", ["read_data_blocksize", "read_data_fpp"])
@pytest.mark.parametrize(
    "input_meta", [{"id": "str"}, {"text": "str"}, {"id": "str", "text": "str"}]
)
def test_read_data_input_meta(
    mock_multiple_jsonl_files, backend, function_name, input_meta
):
    if function_name == "read_data_fpp":
        func = read_data_fpp
        read_kwargs = {"files_per_partition": 1}
    elif function_name == "read_data_blocksize":
        func = read_data_blocksize
        read_kwargs = {"blocksize": "128Mib"}

    df = func(
        input_files=mock_multiple_jsonl_files,
        backend=backend,
        file_type="jsonl",
        add_filename=False,
        input_meta=input_meta,
        columns=None,
        **read_kwargs,
    )

    if function_name == "read_data_fpp" and backend == "cudf":
        assert list(df.columns) == list(input_meta.keys())
    else:
        # In the read_data_fpp case, because pandas doesn't support `prune_columns`, it'll always return all columns even if input_meta is specified
        # In the `read_data_blocksize` case, `dask.read_json` also doesn't `prune_columns` so it'll always return all columns
        # if you user wants to select subset of columns, they should use `columns` parameter
        assert list(df.columns) == ["id", "text"]


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("file_type", ["parquet", "jsonl"])
@pytest.mark.parametrize(
    "read_kwargs",
    [
        *[({"files_per_partition": fpp, "blocksize": None}) for fpp in range(1, 6)],
        *[
            ({"blocksize": bs, "files_per_partition": None})
            for bs in
            #   ["1kb", "5kb", "10kb"]
            ["128MiB", "256MiB", "512MiB"]
        ],
    ],
)
def test_read_data_different_columns(
    mock_multiple_jsonl_files_different_cols,
    mock_multiple_parquet_files_different_cols,
    backend,
    file_type,
    read_kwargs,
):

    read_kwargs_cp = read_kwargs.copy()
    # if function_name == "read_data_fpp":
    #     func = read_data_fpp
    #     # read_kwargs = {"files_per_partition": 2}
    # elif function_name == "read_data_blocksize":
    #     func = read_data_blocksize
    #     # read_kwargs = {"blocksize": "1kb"}

    read_kwargs_cp["columns"] = ["adlr_id", "text"]
    random.seed(0)
    if file_type == "jsonl":
        # input_files = mock_multiple_jsonl_files_different_cols
        input_files = random.choices(
            get_all_files_paths_under("/raid/prospector-lm/rpv1_json/"), k=10
        )

        # read_kwargs_cp["input_meta"] = {"id": "str", "text": "str"}
        # read_kwargs_cp["meta"] = {"id": "str", "text": "str"}

    else:
        # input_files = mock_multiple_parquet_files_different_cols
        input_files = random.choices(
            get_all_files_paths_under("/raid/prospector-lm/rpv1_parquet/"), k=10
        )
        if backend == "cudf":
            read_kwargs_cp["allow_mismatched_pq_schemas"] = True

    df = read_data(
        input_files=input_files,
        file_type=file_type,
        backend=backend,
        add_filename=False,
        **read_kwargs_cp,
    )
    assert list(df.columns) == ["adlr_id", "text"]
    assert list(df.compute().columns) == ["adlr_id", "text"]
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_parquet(tmpdir)
    # assert len(df) == NUM_FILES * NUM_RECORDS
