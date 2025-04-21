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
