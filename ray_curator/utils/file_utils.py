import warnings
from multiprocessing import Pool

import fsspec
import pyarrow.parquet as pq
from fsspec.core import get_filesystem_class, split_protocol


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


def get_parquet_uncompressed_size(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get uncompressed size for local/cloud Parquet files"""
    with fsspec.open(file_path, "rb", **(storage_options or {})) as f:
        parquet_file = pq.ParquetFile(f)
        return sum(
            parquet_file.metadata.row_group(rg).total_byte_size for rg in range(parquet_file.metadata.num_row_groups)
        )


def get_parquet_num_rows(
    file_path: str,
    storage_options: dict | None = None,
) -> int:
    """Get number of rows for local/cloud Parquet files"""
    with fsspec.open(file_path, "rb", **(storage_options or {})) as f:
        parquet_file = pq.ParquetFile(f)
        return parquet_file.metadata.num_rows


def _get_file_size(file: str, storage_options: dict | None = None) -> tuple[str, int]:
    return file, get_parquet_uncompressed_size(file, storage_options)


def split_parquet_files_into_chunks(files: list[str], n: int, storage_options: dict | None = None) -> list[list[str]]:
    """Split files into N chunks with balanced uncompressed sizes."""

    with Pool() as pool:
        sizes = pool.starmap(_get_file_size, [(file, storage_options) for file in files])

    chunks = [[] for _ in range(n)]
    chunk_sizes = [0] * n

    for file, size in sorted(sizes, key=lambda x: -x[1]):
        min_index = chunk_sizes.index(min(chunk_sizes))
        chunks[min_index].append(file)
        chunk_sizes[min_index] += size

    return chunks


def get_embedding_dim_from_file(
    file: str,
    embedding_col: str,
    storage_options: dict | None = None,
) -> int:
    """
    Get the embedding dimension from a single parquet file and embedding column.
    """
    with fsspec.open(file, "rb", **(storage_options or {})) as f:
        parquet_file = pq.ParquetFile(f)
        schema = parquet_file.schema_arrow
        if embedding_col in schema.names:
            field = schema.field(embedding_col)
            if (
                hasattr(field.type, "value_type")
                and hasattr(field.type, "list_size")
                and field.type.list_size is not None
            ):
                return field.type.list_size
            else:
                table = parquet_file.read_row_groups([0], columns=[embedding_col], use_threads=False)
                arr = table[embedding_col][0]
                return len(arr) if hasattr(arr, "__len__") else 1
        else:
            return 1


def split_files_by_max_elements(
    files: list[str],
    embedding_dim: int,
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


def split_files_into_microbatches(
    files: list[str],
    embedding_col: str,
    max_total_elements: int = 2_000_000_000,
    storage_options: dict | None = None,
) -> list[list[str]]:
    """
    Backward-compatible convenience function: gets embedding_dim from first file, then splits files.
    """
    if not files:
        return []
    embedding_dim = get_embedding_dim_from_file(files[0], embedding_col, storage_options)
    return split_files_by_max_elements(files, embedding_dim, max_total_elements, storage_options)
