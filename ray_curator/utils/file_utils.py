import warnings
from multiprocessing import Pool
from typing import Literal

import fsspec
import pyarrow.parquet as pq
from fsspec.core import get_filesystem_class, split_protocol
from fsspec.parquet import open_parquet_file
from loguru import logger


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


def filter_files_by_extension(
    files_list: list[str],
    keep_extensions: str | list[str],
) -> list[str]:
    filtered_files = []
    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]

    # Ensure that the extensions are prefixed with a dot
    file_extensions = tuple([s if s.startswith(".") else f".{s}" for s in keep_extensions])

    for file in files_list:
        if file.endswith(file_extensions):
            filtered_files.append(file)

    if len(files_list) != len(filtered_files):
        warnings.warn("Skipped at least one file due to unmatched file extension(s).", stacklevel=2)

    return filtered_files


def get_all_files_paths_under(
    input_dir: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    # TODO: update with a more robust fsspec method
    if fs is None:
        fs = get_fs(input_dir, storage_options)

    file_ls = fs.find(input_dir, maxdepth=None if recurse_subdirectories else 1)
    if "://" in input_dir:
        protocol = input_dir.split("://")[0]
        file_ls = [f"{protocol}://{f}" for f in file_ls]

    file_ls.sort()
    if keep_extensions is not None:
        file_ls = filter_files_by_extension(file_ls, keep_extensions)
    return file_ls


def parse_bytes(s: float | str) -> int:
    """Parse byte string to numbers. Copied from dask.utils.parse_bytes

    >>> parse_bytes('100')
    100
    >>> parse_bytes('100 MB')
    100000000
    >>> parse_bytes('100M')
    100000000
    >>> parse_bytes('5kB')
    5000
    >>> parse_bytes('5.4 kB')
    5400
    >>> parse_bytes('1kiB')
    1024
    >>> parse_bytes('1e6')
    1000000
    >>> parse_bytes('1e6 kB')
    1000000000
    >>> parse_bytes('MB')
    1000000
    >>> parse_bytes(123)
    123
    >>> parse_bytes('5 foos')
    Traceback (most recent call last):
        ...
    ValueError: Could not interpret 'foos' as a byte unit
    """
    byte_sizes = {
        "kB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
        "PB": 10**15,
        "KiB": 2**10,
        "MiB": 2**20,
        "GiB": 2**30,
        "TiB": 2**40,
        "PiB": 2**50,
        "B": 1,
        "": 1,
    }
    byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
    byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and "i" not in k})
    byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and "i" in k})
    if isinstance(s, (int, float)):
        return int(s)
    s = s.replace(" ", "")
    if not any(char.isdigit() for char in s):
        s = "1" + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    try:
        n = float(prefix)
    except ValueError as e:
        msg = f"Could not interpret '{prefix}' as a number"
        raise ValueError(msg) from e

    try:
        multiplier = byte_sizes[suffix.lower()]
    except KeyError as e:
        msg = f"Could not interpret '{suffix}' as a byte unit"
        raise ValueError(msg) from e

    result = n * multiplier
    return int(result)


def get_parquet_uncompressed_size(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get uncompressed size for local/cloud Parquet files efficiently by only reading the footer"""
    with open_parquet_file(file_path, storage_options=storage_options) as f:
        metadata = pq.read_metadata(f)
        return sum(metadata.row_group(rg).total_byte_size for rg in range(metadata.num_row_groups))


def get_parquet_num_rows(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get number of rows for local/cloud Parquet files efficiently by only reading the footer"""
    with open_parquet_file(file_path, storage_options=storage_options) as f:
        return pq.read_metadata(f).num_rows


def _get_pq_file_size(file: str, storage_options: dict | None = None) -> tuple[str, int]:
    return file, get_parquet_uncompressed_size(file, storage_options)


def _split_files_as_per_fpp_or_num_partitions(
    sorted_file_sizes: list[tuple[str, int]],
    files_per_partition: int | None = None,
    num_partitions: int | None = None,
) -> list[list[str]]:
    n = len(sorted_file_sizes) // files_per_partition if files_per_partition is not None else num_partitions
    partitions = [[] for _ in range(n)]
    partition_sizes = [0] * n

    for file, size in sorted_file_sizes:
        min_index = partition_sizes.index(min(partition_sizes))
        partitions[min_index].append(file)
        partition_sizes[min_index] += size
    logger.info(
        f"Split {len(sorted_file_sizes)} files into {n} partitions"
        + (f" with ~{files_per_partition} fpp" if files_per_partition is not None else "")
    )
    return partitions


def _split_files_as_per_blocksize(
    sorted_file_sizes: list[tuple[str, int]], max_byte_per_chunk: int
) -> list[list[str]]:
    partitions = []
    current_partition = []
    current_size = 0

    for file, size in sorted_file_sizes:
        if current_size + size > max_byte_per_chunk:
            partitions.append(current_partition)
            current_partition = []
            current_size = 0
        current_partition.append(file)
        current_size += size
    if current_partition:
        partitions.append(current_partition)

    logger.info(
        f"Split {len(sorted_file_sizes)} files into {len(partitions)} partitions with max size {(max_byte_per_chunk / 1024 / 1024):.2f} MB."
    )
    return partitions


def _split_files_as_per_read_approach(
    sorted_file_sizes: list[tuple[str, int]],
    read_approach: Literal["blocksize", "fpp", "min_partitions"],
    read_size: int,
) -> list[list[str]]:
    if read_approach == "fpp":
        return _split_files_as_per_fpp_or_num_partitions(
            sorted_file_sizes, files_per_partition=read_size, num_partitions=None
        )
    elif read_approach == "min_partitions":
        return _split_files_as_per_fpp_or_num_partitions(
            sorted_file_sizes, files_per_partition=None, num_partitions=read_size
        )
    elif read_approach == "blocksize":
        return _split_files_as_per_blocksize(sorted_file_sizes, read_size)
    else:
        msg = f"Invalid read approach: {read_approach}"
        raise ValueError(msg)


def smart_split_files_as_per_read_approach(
    sorted_file_sizes: list[tuple[str, int]],
    read_approach: Literal["blocksize", "fpp"],
    read_size: str | int,
    min_number_of_partitions: int | None = None,
) -> list[list[str]]:
    if read_approach not in {"blocksize", "fpp"}:
        msg = f"Smart splitting is not supported for read approach: {read_approach}"
        raise ValueError(msg)
    read_size = parse_bytes(read_size) if isinstance(read_size, str) else read_size

    partitions = _split_files_as_per_read_approach(sorted_file_sizes, read_approach, read_size)

    # Perform binary search to ensure that number of partitions is atleast equal to number of workers
    # Following the binary search template, where we find the first K satisfying the condition
    # Here are condition is that number of partitions should be atleast equal to number of workers
    if len(partitions) <= min_number_of_partitions:
        min_increment = 1 if read_approach == "fpp" else 1024**2  # 1MB for blocksize
        left = min_increment
        right = read_size
        while left < right:
            mid = left + (right - left) // 2
            partitions = _split_files_as_per_read_approach(sorted_file_sizes, read_approach, mid)
            if len(partitions) <= min_number_of_partitions:
                right = mid
            else:
                left = mid + min_increment

        partitions = _split_files_as_per_read_approach(sorted_file_sizes, read_approach, left)

    return partitions


def split_parquet_files_into_chunks(
    files: list[str],
    read_approach: Literal["blocksize", "fpp", "min_partitions"],
    read_size: str | int | None = None,
    min_number_of_partitions: int | None = None,
    storage_options: dict | None = None,
) -> list[list[str]]:
    """
    Args:
        files: list of parquet files to split
        read_approach: approach to read the files. one of blocksize, fpp, min_partitions
        read_size: size to read the files, required for fpp and blocksize
        min_number_of_partitions: minimum number of partitions to return; ensures that number of partitions is atleast equal to number of workers. if none then num_gpus is assumed
        storage_options: storage options for the files
    """
    from ray_curator.utils.ray_utils import get_num_gpus

    if read_approach in {"blocksize", "fpp", "min_partitions"}:
        min_number_of_partitions = get_num_gpus() if min_number_of_partitions is None else min_number_of_partitions
        if len(files) <= min_number_of_partitions:
            # In case we have fewer files than workers, we atleast return one task per file and not coalesce multiple files per task
            return [[file] for file in files]
        # When we have more files than workers, we try to give each worker a fair share of files so we calculate sizes
        with Pool() as pool:
            file_sizes = pool.starmap(_get_pq_file_size, [(file, storage_options) for file in files])
        # Sort files by size in descending order
        file_sizes.sort(key=lambda x: -x[1])
        if read_approach == "min_partitions":
            return _split_files_as_per_fpp_or_num_partitions(
                file_sizes, files_per_partition=None, num_partitions=min_number_of_partitions
            )
        else:
            return smart_split_files_as_per_read_approach(
                file_sizes, read_approach, read_size, min_number_of_partitions
            )
    else:
        msg = f"Invalid read approach: {read_approach}"
        raise ValueError(msg)


def get_embedding_dim_from_pq_file(
    file: str,
    storage_options: dict | None = None,
) -> int:
    """
    Get the total embedding dimension by summing lengths of all list[X] columns.
    Assumes all rows in each list column have the same length.
    """
    with fsspec.open(file, "rb", **(storage_options or {})) as f:
        parquet_file = pq.ParquetFile(f)
        schema = parquet_file.schema_arrow

        total_dim = 0
        # Read first row group to get actual lengths
        table = parquet_file.read_row_groups([0], use_threads=False)

        for field in schema:
            # Check if field is a list type
            if hasattr(field.type, "value_type"):
                # Get length from first row
                arr = table[field.name][0]
                if hasattr(arr, "__len__"):
                    total_dim += len(arr)

        if total_dim == 0:
            # If no list columns found, or all list columns are empty, return 0
            return 0

        return total_dim


def split_pq_files_by_max_elements(
    files: list[str],
    embedding_dim: int | None = None,
    max_total_elements: int = 2_000_000_000,
    storage_options: dict | None = None,
) -> list[list[str]]:
    """
    Split files into microbatches such that the total number of elements in the embedding column
    (sum_rows * embedding_dim) in each microbatch does not exceed max_total_elements.
    Uses a simple greedy bin-packing based on cumulative row count.
    """
    # Get num_rows for each file up front
    num_rows_list = [get_parquet_num_rows(file, storage_options) for file in files]
    if embedding_dim is None:
        embedding_dim = get_embedding_dim_from_pq_file(files[0], storage_options)

    microbatches = []
    current_batch = []
    current_rows = 0
    for file, num_rows in zip(files, num_rows_list, strict=False):
        if (current_rows + num_rows) * embedding_dim > max_total_elements and current_batch:
            microbatches.append(current_batch)
            current_batch = []
            current_rows = 0
        current_batch.append(file)
        current_rows += num_rows
    if current_batch:
        microbatches.append(current_batch)
    return microbatches
