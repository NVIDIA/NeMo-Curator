# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import ast
import os
import shutil
import socket
import subprocess
import uuid
from collections import Counter, defaultdict

import dask

from nemo_curator._compat import DASK_CUDF_PARQUET_READ_INCONSISTENT_SCHEMA

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import random
import warnings
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
from dask.distributed import Client, LocalCluster, get_worker, performance_report
from distributed.diagnostics.nvml import has_cuda_context

from nemo_curator.utils.gpu_utils import is_cudf_type
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")
get_device_total_memory = gpu_only_import_from("dask_cuda.utils", "get_device_total_memory")
if TYPE_CHECKING:
    from collections.abc import Callable

SUPPORTED_JSONL_COMPRESSIONS = {"gzip", None}


class NoWorkerError(Exception):
    pass


def _enable_spilling() -> None:
    """
    Setting this environment variable enables automatic spilling (and "unspilling")
    of buffers from device to host to enable out-of-memory computation,
    i.e., computing on objects that occupy more memory than is available on the GPU.
    """
    # Workaround for below (which is missing in 24.08, but fixed in 24.10)
    # Remove this when we update to 24.10 or later dask-cuda
    # https://github.com/rapidsai/dask-cuda/pull/1369/files
    cudf.set_option("spill", True)


def get_filepath_without_extension(path: str) -> str:
    known_suffixes = [".jsonl", ".json", ".parquet", ".gz", ".part", ".snappy"]
    p = Path(path)
    filename = p.name
    for s in reversed(p.suffixes):
        if s in known_suffixes:
            filename = filename.removesuffix(s)
        else:
            # Exit loop if we encounter an unknown suffix
            break
    return filename


def _worker_gpu_tuple() -> tuple[str, int]:
    """
    Runs on a Dask-CUDA worker.
    Returns (hostname, gpu_index) where `gpu_index` is the index shown by `nvidia-smi`.
    """

    # Touch the GPU so a context is created (idempotent if one already exists)
    cp.cuda.runtime.getDevice()
    ctx = has_cuda_context()

    # Robust hostname lookup
    try:
        hostname = socket.gethostname()
    except Exception as exc:  # noqa: BLE001  (broad on purpose)
        placeholder = f"unknown-{uuid.uuid4().hex[:8]}"
        warnings.warn(
            f"socket.gethostname() failed: {exc!r}. Using placeholder host name '{placeholder}'.",
            stacklevel=2,
        )
        hostname = placeholder
    if ctx.has_context and ctx.device_info is not None:
        device_index = ctx.device_info.device_index
    else:
        # Fallback - context not yet created or NVML unavailable
        warnings.warn(
            f"Unable to retrieve valid GPU index for host '{hostname}'. "
            "GPU context may not be initialized or NVML may be unavailable. Returning -1.",
            stacklevel=2,
        )
        device_index = -1
    return hostname, device_index


def _assert_unique_gpu_per_host(client: Client) -> None:
    """
    Verifies that each Dask worker on a given host is bound to a unique GPU.

    Raises
    ------
    RuntimeError
        If two or more workers on the same host are bound to the same GPU.
        The error message details:
        • host name
        • GPU index with duplicates
        • number of workers bound to that GPU
        • total workers detected on the host
    """
    # Returns a dictionary of worker addresses to (hostname, gpu_index)
    info = client.run(_worker_gpu_tuple)

    # Group GPU indices by host
    per_host: dict[str, list[int]] = defaultdict(list)
    for host, gpu in info.values():
        per_host[host].append(gpu)

    # Build a human-readable report of duplicates
    duplicate_hosts: list[str] = []
    for host, gpus in per_host.items():
        counts = Counter(gpus)
        # Keep only GPUs bound more than once
        dup_gpus = {gpu: n for gpu, n in counts.items() if n > 1}
        if dup_gpus:
            lines = [f"  GPU {gpu} → {n} workers" for gpu, n in sorted(dup_gpus.items())]
            summary = f"\nHost: {host}  (total workers: {len(gpus)})\n" + "\n".join(lines)
            duplicate_hosts.append(summary)

    if duplicate_hosts:
        report = (
            "Duplicate GPU assignment detected!\n"
            + "\n".join(duplicate_hosts)
            + "\nEach worker on a host must own a distinct GPU."
        )
        raise RuntimeError(report)


def start_dask_gpu_local_cluster(  # noqa: PLR0913
    nvlink_only: bool = False,
    protocol: str = "tcp",
    rmm_pool_size: int | str | None = "1024M",
    enable_spilling: bool = True,
    set_torch_to_use_rmm: bool = True,
    rmm_async: bool = True,
    rmm_maximum_pool_size: int | str | None = None,
    rmm_managed_memory: bool = False,
    rmm_release_threshold: int | str | None = None,
    **cluster_kwargs,
) -> Client:
    """
    This function sets up a Dask cluster across all the
    GPUs present on the machine.

    See get_client function for parameters.

    """
    extra_kwargs = (
        {
            "enable_tcp_over_ucx": True,
            "enable_nvlink": True,
            "enable_infiniband": False,
            "enable_rdmacm": False,
        }
        if nvlink_only and protocol == "ucx"
        else {}
    )

    cluster = LocalCUDACluster(
        rmm_pool_size=rmm_pool_size,
        protocol=protocol,
        enable_cudf_spill=enable_spilling,
        rmm_async=rmm_async,
        rmm_maximum_pool_size=rmm_maximum_pool_size,
        rmm_managed_memory=rmm_managed_memory,
        rmm_release_threshold=rmm_release_threshold,
        **extra_kwargs,
        **cluster_kwargs,
    )
    client = Client(cluster)

    if set_torch_to_use_rmm:
        _set_torch_to_use_rmm()
        client.run(_set_torch_to_use_rmm)
        print("Torch is using RMM memory pool", flush=True)

    if enable_spilling:
        _enable_spilling()
        print("cuDF Spilling is enabled", flush=True)

    if get_num_workers(client) <= 0:
        msg = "No workers are currently connected."
        raise NoWorkerError(msg)
    return client


def start_dask_cpu_local_cluster(
    n_workers: int | None = os.cpu_count(),
    threads_per_worker: int = 1,
    **cluster_kwargs,
) -> Client:
    """
    This function sets up a Dask cluster across all the
    CPUs present on the machine.

    See get_client function for parameters.

    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        **cluster_kwargs,
    )

    client = Client(cluster)
    if get_num_workers(client) <= 0:
        msg = "No workers are currently connected."
        raise NoWorkerError(msg)
    return client


def get_client(  # noqa: PLR0913
    cluster_type: Literal["cpu", "gpu"] = "cpu",
    scheduler_address: str | None = None,
    scheduler_file: str | None = None,
    n_workers: int | None = os.cpu_count(),
    threads_per_worker: int = 1,
    nvlink_only: bool = False,
    protocol: Literal["tcp", "ucx"] = "tcp",
    rmm_pool_size: str | int | None = "1024M",
    enable_spilling: bool = True,
    set_torch_to_use_rmm: bool = False,
    rmm_async: bool = True,
    rmm_maximum_pool_size: str | int | None = None,
    rmm_managed_memory: bool = False,
    rmm_release_threshold: str | int | None = None,
    **cluster_kwargs,
) -> Client:
    """
    Initializes or connects to a Dask cluster.
    The Dask cluster can be CPU-based or GPU-based (if GPUs are available).
    The intialization ensures maximum memory efficiency for the GPU by:
        1. Ensuring the PyTorch memory pool is the same as the RAPIDS memory pool. (If `set_torch_to_use_rmm` is True)
        2. Enabling spilling for cuDF. (If `enable_spilling` is True)

    Args:
        cluster_type: If scheduler_address and scheduler_file are None, sets up a local (single node) cluster of the specified type.
            Either "cpu" or "gpu". Defaults to "cpu". Many options in get_client only apply to CPU-based or GPU-based clusters.
            Make sure you check the description of the parameter.
        scheduler_address: Address of existing Dask cluster to connect to. This can be the address of a Scheduler server like a
            string '127.0.0.1:8786' or a cluster object like LocalCluster(). If specified, all other arguments are ignored and
            the client is connected to the existing cluster. The other configuration options should be done when setting up the
            Dask cluster.
        scheduler_file: Path to a file with scheduler information if available. If specified, all other arguments are ignored
            and the client is connected to the existing cluster. The other configuration options should be done when setting up the
            Dask cluster.
        n_workers: For CPU-based clusters only. The number of workers to start. Defaults to os.cpu_count(). For GPU-based clusters,
            the number of workers is locked to the number of GPUs in CUDA_VISIBLE_DEVICES.
        threads_per_worker: For CPU-based clusters only. The number of threads per each worker. Defaults to 1.
            Before increasing, ensure that your functions frequently release the GIL.
        nvlink_only: For GPU-based clusters only. Whether to use nvlink or not.
        protocol: For GPU-based clusters only. Protocol to use for communication. "tcp" or "ucx".
        rmm_pool_size: For GPU-based clusters only. RMM pool size to initialize each worker with. Can be an integer (bytes),
            float (fraction of total device memory), string (like "5GB" or "5000M"), or None to disable RMM pools.
        enable_spilling: For GPU-based clusters only. Enables automatic spilling (and "unspilling") of buffers from device to
            host to enable out-of-memory computation, i.e., computing on objects that occupy more memory than is available on the GPU.
        set_torch_to_use_rmm: For GPU-based clusters only. Sets up the PyTorch memory pool to be the same as the RAPIDS memory pool.
            This helps avoid OOM errors when using both PyTorch and RAPIDS on the same GPU.
        rmm_async: For GPU-based clusters only. Initializes each worker with RAPIDS Memory Manager (RMM)
            (see RMM documentation for more information: https://docs.rapids.ai/api/rmm/stable/)
            and sets it to use RMM's asynchronous allocator. Warning: The asynchronous allocator requires CUDA Toolkit 11.2 or newer.
            It is also incompatible with RMM pools and managed memory. Trying to enable both will result in an exception.
        rmm_maximum_pool_size: For GPU-based clusters only. When rmm_pool_size is set, this argument indicates the maximum pool size.
            Can be an integer (bytes), float (fraction of total device memory), string (like "5GB" or "5000M") or None.
            By default, the total available memory on the GPU is used.
            rmm_pool_size must be specified to use RMM pool and to set the maximum pool size.
            Note: When paired with --enable-rmm-async the maximum size cannot be guaranteed due to fragmentation.
            Note: This size is a per-worker configuration, and not cluster-wide.
        rmm_managed_memory: For GPU-based clusters only. Initialize each worker with RMM and set it to use managed memory.
            If disabled, RMM may still be used by specifying rmm_pool_size.
            Warning: Managed memory is currently incompatible with NVLink. Trying to enable both will result in an exception.
        rmm_release_threshold: For GPU-based clusters only. When rmm.async is True and the pool size grows beyond this value,
            unused memory held by the pool will be released at the next synchronization point.
            Can be an integer (bytes), float (fraction of total device memory), string (like "5GB" or "5000M") or None.
            By default, this feature is disabled.
            Note: This size is a per-worker configuration, and not cluster-wide.
        cluster_kwargs: Additional keyword arguments for the LocalCluster or LocalCUDACluster configuration.
            See API documentation https://docs.dask.org/en/stable/deploying-python.html#distributed.deploy.local.LocalCluster
            for all LocalCluster parameters, or https://docs.rapids.ai/api/dask-cuda/nightly/api/ for all LocalCUDACluster parameters.
    Returns:
        A Dask client object.

    """
    if cluster_type not in ["cpu", "gpu"]:
        msg = f"Unknown cluster type: {cluster_type}"
        raise ValueError(msg)

    if scheduler_address:
        if scheduler_file:
            msg = "Only one of scheduler_address or scheduler_file can be provided"
            raise ValueError(msg)
        else:
            client = Client(address=scheduler_address, timeout="30s")
            if get_num_workers(client) <= 0:
                msg = "No workers are currently connected."
                raise NoWorkerError(msg)
    elif scheduler_file:
        client = Client(scheduler_file=scheduler_file, timeout="30s")
        if get_num_workers(client) <= 0:
            msg = "No workers are currently connected."
            raise NoWorkerError(msg)
    elif cluster_type == "gpu":
        client = start_dask_gpu_local_cluster(
            nvlink_only=nvlink_only,
            protocol=protocol,
            rmm_pool_size=rmm_pool_size,
            enable_spilling=enable_spilling,
            set_torch_to_use_rmm=set_torch_to_use_rmm,
            rmm_async=rmm_async,
            rmm_maximum_pool_size=rmm_maximum_pool_size,
            rmm_managed_memory=rmm_managed_memory,
            rmm_release_threshold=rmm_release_threshold,
            **cluster_kwargs,
        )
    else:
        client = start_dask_cpu_local_cluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            **cluster_kwargs,
        )
    if cluster_type == "gpu":
        _assert_unique_gpu_per_host(client)
    return client


def _set_torch_to_use_rmm() -> None:
    """
    This function sets up the PyTorch memory pool to be the same as the RAPIDS memory pool.
    This helps avoid OOM errors when using both PyTorch and RAPIDS on the same GPU.

    See article:
    https://medium.com/rapids-ai/pytorch-rapids-rmm-maximize-the-memory-efficiency-of-your-workflows-f475107ba4d4
    """

    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    if torch.cuda.get_allocator_backend() == "pluggable":
        warnings.warn(
            "PyTorch allocator already plugged in, not switching to RMM. "
            "Please ensure you have not already swapped it.",
            stacklevel=2,
        )
        return

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def _resolve_filename_col(filename: bool | str) -> str | bool:
    if filename is False:
        return False
    elif filename is True:
        return "file_name"
    elif isinstance(filename, str):
        return filename
    else:
        msg = f"Unknown filename value: {filename}"
        raise ValueError(msg)


def select_columns(
    df: dd.DataFrame | pd.DataFrame | cudf.DataFrame,
    columns: list[str],
    file_type: Literal["jsonl", "json", "parquet"],
    add_filename: bool | str,
) -> dd.DataFrame | pd.DataFrame | cudf.DataFrame:
    # We exclude parquet because the parquet readers already support column selection
    if file_type in ["jsonl", "json"] and columns is not None:
        if add_filename:
            filename_str = _resolve_filename_col(add_filename)
            if filename_str not in columns:
                columns.append(filename_str)
        df = df[columns]

    return df


def read_single_partition(  # noqa: C901, PLR0912, PLR0913
    files: list[str],
    backend: Literal["cudf", "pandas"] = "cudf",
    file_type: str = "jsonl",
    add_filename: bool | str = False,
    input_meta: str | dict | None = None,
    io_columns: list[str] | None = None,
    **kwargs,
) -> cudf.DataFrame | pd.DataFrame:
    """
    This function reads a file with cuDF, sorts the columns of the DataFrame
    and adds a filename column.

    Args:
        files: The path to the jsonl files to read.
        backend: The backend to use for reading the data. Either "cudf" or "pandas".
        add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
        file_type: The type of the file to read.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.
        columns: If not None, only these columns will be read from the file.
            There is a significant performance gain when specifying columns for Parquet files.

    Returns:
        A cudf DataFrame or a pandas DataFrame.

    """
    if input_meta is not None and file_type != "jsonl":
        warnings.warn(
            "input_meta is only valid for JSONL files and will be ignored for other  file formats.",
            stacklevel=2,
        )

    if file_type in ["jsonl", "json"]:
        read_kwargs = {"lines": file_type == "jsonl"}
        if backend == "cudf":
            read_f = cudf.read_json
            if input_meta is not None:
                read_kwargs["prune_columns"] = True
        else:
            read_kwargs["dtype"] = False
            read_f = pd.read_json

        if input_meta is not None:
            read_kwargs["dtype"] = ast.literal_eval(input_meta) if isinstance(input_meta, str) else input_meta
            # because pandas doesn't support `prune_columns`, it'll always return all columns even when input_meta is specified
            # to maintain consistency we explicitly set `io_columns` here
            if backend == "pandas" and not io_columns:
                io_columns = list(read_kwargs["dtype"].keys())

    elif file_type == "parquet":
        read_kwargs = {"columns": io_columns}
        read_f = cudf.read_parquet if backend == "cudf" else pd.read_parquet

    else:
        msg = f"Could not read data, please check file type: {file_type}"
        raise RuntimeError(msg)

    if add_filename:
        read_files_one_at_a_time = True
    elif backend == "cudf":
        # cuDF supports reading multiple files at once
        read_files_one_at_a_time = False
    else:
        # Pandas does not support reading multiple files at once
        read_files_one_at_a_time = True

    if read_files_one_at_a_time:
        concat_f = cudf.concat if backend == "cudf" else pd.concat
        df_ls = []
        for file in files:
            df = read_f(file, **read_kwargs, **kwargs)
            if add_filename:
                df[_resolve_filename_col(add_filename)] = os.path.basename(file)
            df = select_columns(df, io_columns, file_type, add_filename)
            df_ls.append(df)

        df = concat_f(df_ls, ignore_index=True)
    else:
        df = read_f(files, **read_kwargs, **kwargs)
        df = select_columns(df, io_columns, file_type, add_filename)
    return df


def read_data_blocksize(  # noqa: C901, PLR0913
    input_files: list[str],
    backend: Literal["cudf", "pandas"],
    file_type: Literal["parquet", "jsonl"],
    blocksize: str,
    add_filename: bool | str = False,
    input_meta: str | dict | None = None,
    columns: list[str] | None = None,
    **kwargs,
) -> dd.DataFrame:
    read_kwargs = {}

    if file_type == "jsonl":
        warnings.warn(
            "If underlying JSONL data does not have a consistent schema, reading with blocksize will fail. "
            "Please use files_per_partition approach instead.",
            stacklevel=2,
        )

        if backend == "pandas":
            warnings.warn(
                "Pandas backend with blocksize cannot read multiple JSONL files into a single partition. "
                "Please use files_per_partition if blocksize exceeds average file size.",
                stacklevel=2,
            )
        read_func = dd.read_json
        read_kwargs["lines"] = True
        if input_meta is not None:
            if backend == "cudf":
                # To save GPU memory, we prune columns while reading, and keep only those that are
                # specified in the input_meta
                read_kwargs["prune_columns"] = True

            read_kwargs["dtype"] = ast.literal_eval(input_meta) if isinstance(input_meta, str) else input_meta

            if not columns:
                # To maintain consistency with the behavior of `read_data_fpp` where passing `input_meta`
                # only returns those columns, we explicitly set `columns` here
                columns = list(read_kwargs["dtype"].keys())
        if add_filename:

            def extract_filename(path: str) -> str:
                return os.path.basename(path)

            read_kwargs["include_path_column"] = _resolve_filename_col(add_filename)
            read_kwargs["path_converter"] = extract_filename

    elif file_type == "parquet":
        if backend == "cudf" and not DASK_CUDF_PARQUET_READ_INCONSISTENT_SCHEMA:
            warnings.warn(
                "If underlying Parquet data does not have consistent schema, reading with blocksize will fail. "
                "Please update underlying RAPIDS package to version 25.02 or higher, or use files_per_partition approach instead.",
                stacklevel=2,
            )
        elif backend == "pandas":
            warnings.warn(
                "If underlying Parquet data does not have a consistent column order, reading with blocksize might fail. "
                "Please use files_per_partition approach instead.",
                stacklevel=2,
            )

        if add_filename:
            msg = "add_filename and blocksize cannot be set at the same time for Parquet files."
            raise ValueError(msg)
        read_func = dd.read_parquet
        read_kwargs["columns"] = columns
        # In dask_cudf >= 24.12, aggregate_files is not required, but we've kept here until
        # it gets in dask (pandas) as well
        read_kwargs["aggregate_files"] = True
    else:
        msg = f"Reading with blocksize is only supported for JSONL and Parquet files, not {file_type=}"
        raise ValueError(msg)

    with dask.config.set({"dataframe.backend": backend}):
        df = read_func(input_files, blocksize=blocksize, **read_kwargs, **kwargs)

        output = select_columns(df, columns, file_type, add_filename)
        return output[sorted(output.columns)]


def read_data_files_per_partition(  # noqa: PLR0913
    input_files: list[str],
    file_type: Literal["parquet", "json", "jsonl"],
    backend: Literal["cudf", "pandas"] = "cudf",
    add_filename: bool | str = False,
    files_per_partition: int | None = None,
    input_meta: str | dict | None = None,
    columns: list[str] | None = None,
    read_func_single_partition: Callable[[list[str], str, bool, str | dict, dict], dd.DataFrame | pd.DataFrame]
    | None = None,
    **kwargs,
) -> dd.DataFrame:
    if read_func_single_partition is None:
        read_func_single_partition = read_single_partition
        read_func_single_partition_kwargs = dict(
            enforce_metadata=False,
            io_columns=columns,
            **kwargs,
        )
    else:
        read_func_single_partition_kwargs = kwargs

    if files_per_partition > 1:
        input_files = [
            input_files[i : i + files_per_partition] for i in range(0, len(input_files), files_per_partition)
        ]
    else:
        input_files = [[file] for file in input_files]

    output = dd.from_map(
        read_func_single_partition,
        input_files,
        file_type=file_type,
        backend=backend,
        add_filename=add_filename,
        input_meta=input_meta,
        **read_func_single_partition_kwargs,
    )
    return output[sorted(output.columns)]


def read_pandas_pickle(
    file: str,
    add_filename: bool | str = False,
    columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    This function reads a pickle file with Pandas.

    Args:
        file: The path to the pickle file to read.
        add_filename: Whether to add a "file_name" column to the DataFrame.
        columns: If not None, only these columns will be read from the file.
    Returns:
        A Pandas DataFrame.

    """
    if add_filename:
        warnings.warn(
            "add_filename is not supported for pickle files",
            stacklevel=2,
        )

    if columns is not None:
        return pd.read_pickle(file, **kwargs)[columns]  # noqa: S301
    else:
        return pd.read_pickle(file, **kwargs)  # noqa: S301


def read_data(  # noqa: C901, PLR0913
    input_files: str | list[str],
    file_type: str = "pickle",
    backend: Literal["cudf", "pandas"] = "cudf",
    blocksize: str | None = None,
    files_per_partition: int | None = 1,
    add_filename: bool | str = False,
    input_meta: str | dict | None = None,
    columns: list[str] | None = None,
    read_func_single_partition: Callable[[list[str], str, bool, str | dict, dict], dd.DataFrame | pd.DataFrame]
    | None = None,
    **kwargs,
) -> dd.DataFrame:
    """
    This function can read multiple data formats and returns a Dask-cuDF DataFrame.

    Args:
        input_files: The path of the input file(s).
        file_type: The type of the input file(s).
        backend: The backend to use for reading the data.
        blocksize: The size of desired indidivudal partition to be read from files. Either blocksize or files_per_partition must be set.
        files_per_partition: The number of files to read per partition. Either blocksize or files_per_partition must be set.
        add_filename: Whether to add a "file_name" column to the DataFrame.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.
        columns: If not None, only these columns will be read from the file.
            There is a significant performance gain when specifying columns for Parquet files.
        read_func_single_partition: A function that reads a single partition of data.
            This can only be used in conjunction with files_per_partition.
            The function should take the following arguments:
                - files: A list of file paths that will be read in a single partition.
                - file_type: The type of the file to read (in case you want to handle different file types differently).
                - backend: The backend to use for reading the data. (cudf or pandas)
                - add_filename: Read below
                - columns: Read below
                - input_meta: Read below

    Returns:
        A Dask-cuDF or a Dask-pandas DataFrame.

    """
    if isinstance(input_files, str):
        input_files = [input_files]

    check_dask_cwd(input_files)

    if read_func_single_partition is not None and files_per_partition is not None:
        return read_data_files_per_partition(
            input_files,
            file_type=file_type,
            backend=backend,
            add_filename=add_filename,
            files_per_partition=files_per_partition,
            input_meta=input_meta,
            columns=columns,
            read_func_single_partition=read_func_single_partition,
            **kwargs,
        )
    elif file_type == "pickle":
        df = read_pandas_pickle(input_files[0], add_filename=add_filename, columns=columns, **kwargs)
        df = dd.from_pandas(df, npartitions=16)
        if backend == "cudf":
            df = df.to_backend("cudf")

    elif file_type in ["json", "jsonl", "parquet"]:
        if len(input_files) == 0:
            msg = "No input files provided"
            raise ValueError(msg)

        input_extensions = {os.path.splitext(f)[-1] for f in input_files}
        if len(input_extensions) != 1:
            msg = (
                "All files being read must have the same file type. "
                "Please check your input directory or list of files to ensure this. "
                "To generate a list of files with a given file type in your directory, "
                "please use the nemo_curator.utils.file_utils.get_all_files_paths_under "
                "function with the `keep_extensions` parameter."
            )
            raise RuntimeError(msg)

        print(
            f"Reading {len(input_files)} files with {blocksize=} / {files_per_partition=}",
            flush=True,
        )
        if blocksize is not None and files_per_partition is not None:
            msg = "blocksize and files_per_partition cannot be set at the same time"
            raise ValueError(msg)

        if blocksize is not None and (file_type == "jsonl" or (file_type == "parquet" and not add_filename)):
            return read_data_blocksize(
                input_files,
                backend=backend,
                file_type=file_type,
                blocksize=blocksize,
                add_filename=add_filename,
                input_meta=input_meta,
                columns=columns,
                **kwargs,
            )
        else:
            if backend == "cudf" and (file_type == "jsonl" or (file_type == "parquet" and not add_filename)):
                warnings.warn(
                    "Consider passing in blocksize for better control over memory usage.",
                    stacklevel=2,
                )
            return read_data_files_per_partition(
                input_files,
                file_type=file_type,
                backend=backend,
                add_filename=add_filename,
                files_per_partition=files_per_partition,
                input_meta=input_meta,
                columns=columns,
                read_func_single_partition=read_func_single_partition,
                **kwargs,
            )
    else:
        msg = f"Could not read data, please check file type: {file_type}"
        raise RuntimeError(msg)

    return df


def process_batch(load_model_function, load_model_kwargs, run_inference_function, run_inference_kwargs):  # noqa: ANN001, ANN201
    """
    This function loads a model on a Dask worker and then runs inference on a batch of data.

    Args:
        load_model_function: A user-provided function for loading a classifier.
        load_model_kwargs: A dictionary of arguments necessary for `load_model_function`.
        run_inference_function: A user-provided function for running inference, which has a "model" argument.
        run_inference_kwargs: A dictionary of arguments necessary for `run_inference_function`.
    Returns:
        Whatever `run_inference_function` returns, such as a list or tensor of predictions.

    """
    model = load_object_on_worker("model", load_model_function, load_model_kwargs)
    return run_inference_function(**run_inference_kwargs, model=model)


def process_all_batches(  # noqa: ANN201
    loader_valid,  # noqa: ANN001
    load_model_function,  # noqa: ANN001
    load_model_kwargs,  # noqa: ANN001
    run_inference_function,  # noqa: ANN001
    run_inference_kwargs,  # noqa: ANN001
):
    """
    This function iterates over batches of data, loading a model and running inference per batch.

    Args:
        loader_valid: An iterable data object, such as a PyTorch DataLoader.
        load_model_function: A user-provided function for loading a classifier.
        load_model_kwargs: A dictionary of arguments necessary for `load_model_function`.
        run_inference_function: A user-provided function for running inference, which has "model" and "batch" arguments.
        run_inference_kwargs: A dictionary of arguments necessary for `run_inference_function`.
    Returns:
        A tensor of predictions for all batches of the data.

    """
    import torch

    return torch.cat(
        [
            process_batch(
                load_model_function,
                load_model_kwargs,
                run_inference_function,
                dict(run_inference_kwargs, batch=batch),
            )
            for batch in loader_valid
        ]
    )


def single_partition_write_with_filename(  # noqa: C901, PLR0912, PLR0913
    df: pd.DataFrame | cudf.DataFrame,
    output_file_dir: str,
    keep_filename_column: bool = False,
    output_type: str = "jsonl",
    filename_col: str = "file_name",
    compression: str | None = None,
) -> cudf.Series | pd.Series:
    """
    This function processes a DataFrame and writes it to disk

    Args:
        df: A DataFrame.
        output_file_dir: The output file path.
        keep_filename_column: Boolean representing whether to keep or drop the `filename_col`, if it exists.
        output_type: The type of output file to write. Can be "jsonl" or "parquet".
        filename_col: The name of the column that contains the filename. Default is "file_name"
        compression: The compression type to use. Only supported for JSONL files. Can be "gzip" or None
    Returns:
        If the DataFrame is non-empty, return a Series containing a single element, True.
        If the DataFrame is empty, return a Series containing a single element, False.

    """
    if filename_col not in df.columns:
        msg = f"Column {filename_col} not found in DataFrame"
        raise ValueError(msg)

    if len(df) > 0:
        empty_partition = False
    else:
        warnings.warn("Empty partition found", stacklevel=2)
        empty_partition = True

    if is_cudf_type(df):
        import cudf

        success_ser = cudf.Series([empty_partition])
    else:
        success_ser = pd.Series([empty_partition])

    if not empty_partition:
        filenames = df[filename_col].unique()
        filenames = list(filenames.values_host) if is_cudf_type(df) else list(filenames)
        num_files = len(filenames)

        for filename in filenames:
            out_df = df[df[filename_col] == filename] if num_files > 1 else df
            if not keep_filename_column:
                out_df = out_df.drop(filename_col, axis=1)

            filename_without_extension = (
                get_filepath_without_extension(filename) if output_type != "bitext" else Path(filename).name
            )
            output_file_path = os.path.join(output_file_dir, filename_without_extension)

            if output_type == "jsonl":
                output_file_path = output_file_path + ".jsonl"
                if compression not in SUPPORTED_JSONL_COMPRESSIONS:
                    msg = (
                        f"Unsupported compression type: {compression}. Supported types: {SUPPORTED_JSONL_COMPRESSIONS}"
                    )
                    raise ValueError(msg)
                if compression == "gzip":
                    output_file_path = output_file_path + ".gz"

                if isinstance(df, pd.DataFrame):
                    out_df.to_json(
                        output_file_path,
                        orient="records",
                        lines=True,
                        force_ascii=False,
                        index=False,  # Only index=False is supported for orient="records"
                        compression=compression,
                    )
                else:
                    # See open issue here: https://github.com/rapidsai/cudf/issues/15211
                    # df.to_json(output_file_path, orient="records", lines=True, engine="cudf", force_ascii=False)  # noqa: ERA001
                    out_df.to_json(
                        output_file_path,
                        orient="records",
                        lines=True,
                        force_ascii=False,
                        index=False,  # Only index=False is supported for orient="records"
                    )

            elif output_type == "parquet":
                output_file_path = output_file_path + ".parquet"
                out_df.to_parquet(output_file_path)
            elif output_type == "bitext":
                msg = "You shouldn't call this function to write to simple bitext."
                raise RuntimeError(msg)
            else:
                msg = f"Unknown output type: {output_type}"
                raise ValueError(msg)

    return success_ser


def _single_partition_write_to_simple_bitext(
    out_df: cudf.DataFrame | pd.DataFrame, output_file_path: str, partition_info: dict | None = None
) -> cudf.Series | pd.Series:
    if len(out_df) > 0:
        empty_partition = False
    else:
        warnings.warn("Empty partition found", stacklevel=2)
        empty_partition = True

    if is_cudf_type(out_df):
        import cudf

        success_ser = cudf.Series([empty_partition])
    else:
        success_ser = pd.Series([empty_partition])

    # Skip file creation for empty partitions
    if empty_partition:
        return success_ser

    src_output_file_path = output_file_path + f".{out_df['src_lang'].iloc[0]}"
    tgt_output_file_path = output_file_path + f".{out_df['tgt_lang'].iloc[0]}"
    partition_id = partition_info["number"] if partition_info else 0
    with (
        open(f"{src_output_file_path}.{partition_id}", "w") as src_out,
        open(f"{tgt_output_file_path}.{partition_id}", "w") as tgt_out,
    ):
        # Handle cuDF Series which are not directly iterable
        if is_cudf_type(out_df):
            src_values = out_df["src"].to_pandas()
            tgt_values = out_df["tgt"].to_pandas()
        else:
            src_values = out_df["src"]
            tgt_values = out_df["tgt"]

        for src, tgt in zip(src_values, tgt_values, strict=False):
            src_out.write(src + os.linesep)
            tgt_out.write(tgt + os.linesep)

    return success_ser


def _merge_tmp_simple_bitext_partitions(tmp_output_dir: str, output_dir: str) -> None:
    """Merge partitions of simple bitext files in `tmp_output_dir` into files at `output_dir`.

    Args:
        tmp_output_dir (str): temporary directory that has all the simple bitext output partitions,
                with suffixes that looks like "file.1", "file.2" that shows the merging order
        output_file_path (str): dir to write output files
    """

    sorted_tmp_files = sorted(os.listdir(tmp_output_dir), key=lambda x: int(x.split(".")[-1]))
    unique_file_handles = {}
    # Loop through the sorted files and concatenate their contents
    for f in sorted_tmp_files:
        input_file_path = os.path.join(tmp_output_dir, f)
        output_file_name = ".".join(f.split(".")[:-1])

        # this is where current file will be concatenated into
        output_file_path = os.path.join(output_dir, output_file_name)

        # create the output file if we haven't yet
        if output_file_path not in unique_file_handles:
            unique_file_handles[output_file_path] = open(output_file_path, "w")  # noqa: SIM115

        with open(input_file_path) as infile:
            unique_file_handles[output_file_path].write(infile.read())

    # close all dangling file handles
    for handle in unique_file_handles.values():
        handle.close()


def write_to_disk(  # noqa: PLR0912, PLR0913
    df: dd.DataFrame,
    output_path: str,
    write_to_filename: bool | str = False,
    keep_filename_column: bool = False,
    output_type: str = "jsonl",
    partition_on: str | None = None,
    compression: str | None = None,
) -> None:
    """
    This function writes a Dask DataFrame to the specified file path.
    If write_to_filename is True, then it expects the
    DataFrame to have a `filename_col` that specifies where to write the document.

    Args:
        df: A Dask DataFrame.
        output_path: The output file path.
        write_to_filename: Whether to write the filename using the filename column.
                If True the `file_name` column is used to write out.
                If str, uses that as the filename column to write to.
        keep_filename_column: Boolean representing whether to keep or drop the filename column, if it exists.
        output_type: The type of output file to write. Can be "jsonl" or "parquet".
        partition_on: The column name to partition the data on.
                      If specified, the data will be partitioned based on the unique values in this column,
                      and each partition will be written to a separate directory
        compression: The compression type to use. Only supported for JSONL files. Can be "gzip" or None
    """

    filename_col = _resolve_filename_col(write_to_filename)
    # output_path is a file name
    if isinstance(output_path, str) and output_path.endswith(".jsonl"):
        if df.npartitions == 1:
            df.map_partitions(
                _write_to_jsonl_or_parquet,
                output_path,
                output_type,
                compression=compression,
            ).compute()
            return
        else:
            msg = (
                "Could not write multi-partition DataFrame to a single JSONL file. "
                "Please specify a directory output path or repartition the DataFrame."
            )
            raise RuntimeError(msg)

    # output_path is a directory
    elif write_to_filename and filename_col not in df.columns:
        msg = f"write_using_filename is True but no {filename_col} column found in DataFrame"
        raise ValueError(msg)

    if partition_on is not None and write_to_filename:
        msg = "Cannot use both partition_on and write_to_filename parameters simultaneously. "
        raise ValueError(msg)

    if is_cudf_type(df):
        import cudf

        output_meta = cudf.Series([True])
    else:
        output_meta = pd.Series([True], dtype="bool")

    # output_path is a directory
    if write_to_filename and output_type != "bitext":
        os.makedirs(output_path, exist_ok=True)
        output = df.map_partitions(
            single_partition_write_with_filename,
            output_path,
            keep_filename_column=keep_filename_column,
            output_type=output_type,
            filename_col=filename_col,
            compression=compression,
            meta=output_meta,
            enforce_metadata=False,
        )
        output = output.compute()

    # output_path is a directory
    elif output_type in {"jsonl", "parquet"}:
        _write_to_jsonl_or_parquet(
            df,
            output_path=output_path,
            output_type=output_type,
            partition_on=partition_on,
            compression=compression,
        )
    elif output_type == "bitext":
        if write_to_filename:
            os.makedirs(output_path, exist_ok=True)
            tmp_output_file_dir = os.path.join(output_path, ".tmp")
            os.makedirs(tmp_output_file_dir, exist_ok=True)
            file_name = os.path.basename(list(df[filename_col].unique())[0])  # noqa: RUF015
        else:
            tmp_output_file_dir = os.path.join(output_path, ".tmp")
            os.makedirs(tmp_output_file_dir, exist_ok=True)
            file_name = os.path.basename(output_path)

        output = df.map_partitions(
            _single_partition_write_to_simple_bitext,
            os.path.join(tmp_output_file_dir, file_name),
            meta=output_meta,
            enforce_metadata=False,
        )
        output = output.compute()
        _merge_tmp_simple_bitext_partitions(
            tmp_output_file_dir,
            (output_path if write_to_filename else os.path.dirname(output_path)),
        )
        shutil.rmtree(tmp_output_file_dir)
    else:
        msg = f"Unknown output type: {output_type}"
        raise ValueError(msg)

    print(f"Writing to disk complete for {df.npartitions} partition(s)", flush=True)


def _write_to_jsonl_or_parquet(
    df: dd.DataFrame,
    output_path: str,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    partition_on: str | None = None,
    compression: str | None = None,
) -> None:
    if output_type == "jsonl":
        if compression not in SUPPORTED_JSONL_COMPRESSIONS:
            msg = f"Unsupported compression type: {compression}. Supported types: {SUPPORTED_JSONL_COMPRESSIONS}"
            raise ValueError(msg)
        if partition_on is not None:
            unique_values = df[partition_on].unique().to_backend(backend="pandas").compute().to_list()
            for value in unique_values:
                os.makedirs(output_path, exist_ok=True)
                partition_output_path = os.path.join(output_path, f"{partition_on}={value}")
                df[df[partition_on] == value].to_json(
                    partition_output_path,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                    index=False,  # Only index=False is supported for orient="records"
                    compression=compression,
                )
        elif is_cudf_type(df):
            # See open issue here: https://github.com/rapidsai/cudf/issues/15211
            # df.to_json(output_path, orient="records", lines=True, engine="cudf", force_ascii=False)  # noqa: ERA001
            df.to_json(
                output_path,
                orient="records",
                lines=True,
                force_ascii=False,
                index=False,
                compression=compression,
            )  # Only index=False is supported for orient="records"
        else:
            df.to_json(
                output_path,
                orient="records",
                lines=True,
                force_ascii=False,
                index=False,
                compression=compression,
            )  # Only index=False is supported for orient="records"
    elif output_type == "parquet":
        if compression is not None:
            msg = "Setting a custom compression type is not supported for Parquet files at this time."
            raise ValueError(msg)
        df.to_parquet(output_path, write_index=False, partition_on=partition_on)
    else:
        msg = f"Unknown output type: {output_type}"
        raise ValueError(msg)


def load_object_on_worker(attr: str, load_object_function: Callable, load_object_kwargs: dict) -> Any:  # noqa: ANN401
    """
    This function checks if a Dask worker has a specified attribute for storing an object.
    If it does, then fetch the object and return it.
    If it does not, then load the object, set it as an attribute, and return it.

    Args:
        attr: A string representing an attribute of a Dask worker.
        load_object_function: A user-provided function for how to load the object.
        load_object_kwargs: A dictionary of arguments necessary for `load_object_function`.
    Returns:
        The object of interest according to the `attr`.

    """
    # get_worker will fail during type inference of Dask functions
    try:
        worker = get_worker()
    except ValueError as e:
        raise NoWorkerError(str(e))  # noqa: B904

    if hasattr(worker, attr):
        obj = getattr(worker, attr)
    else:
        obj = load_object_function(**load_object_kwargs)
        setattr(worker, attr, obj)
    return obj


def offload_object_on_worker(attr: str) -> bool:
    """
    This function deletes an existing attribute from a Dask worker.

    Args:
        attr: The name of the attribute to delete.
    Returns:
        True.

    """
    worker = get_worker()
    if hasattr(worker, attr):
        delattr(worker, attr)
    return True


def get_num_workers(client: Client | None) -> int | None:
    """
    Returns the number of workers in the cluster
    """
    if client is None:
        return None
    worker_list = list(client.scheduler_info()["workers"].keys())
    return len(worker_list)


def get_current_client() -> Client | None:
    """
    Returns the context-local or latest initailised client.
    If no Client instances exist, returns None
    """
    try:
        return Client.current()
    except ValueError:
        return None


def check_dask_cwd(file_list: list[str]) -> None:
    if any(not os.path.isabs(file_path) for file_path in file_list):
        dask_cwd_list = list(get_current_client().run(os.getcwd).values())
        if len(set(dask_cwd_list)) <= 1:
            dask_cwd = dask_cwd_list[0]
            os_pwd = subprocess.check_output("pwd", shell=True, text=True).strip()  # noqa: S602, S607
            if dask_cwd != os_pwd:
                msg = (
                    "Mismatch between Dask client and worker working directories. "
                    "Use absolute file paths to ensure the correct files are read as intended."
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Mismatch between at least 2 Dask workers' working directories. "
                "Use absolute file paths to ensure the correct files are read as intended."
            )
            raise RuntimeError(msg)


def performance_report_if(
    path: str | None = None, report_name: str = "dask-profile.html"
) -> AbstractContextManager[Any]:
    """
    Generates a performance report if a valid path is provided, or returns a
    no-op context manager if not.

    Args:
        path: The directory path where the performance report should be saved.
            If None, no report is generated.
        report_name: The name of the report file.

    """
    if path is not None:
        return performance_report(os.path.join(path, report_name))
    else:
        return nullcontext()


def performance_report_if_with_ts_suffix(
    path: str | None = None, report_name: str = "dask-profile"
) -> AbstractContextManager[Any]:
    """
    Same as performance_report_if, except it suffixes the report_name with the timestamp.

    """
    return performance_report_if(
        path=path,
        report_name=f"{report_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",  # noqa: DTZ005
    )


def seed_all(seed: int = 42) -> None:
    """
    Function to set seed for random number generators for reproducibility.

    Args:
        seed: The seed value to use for random number generators. Default is 42.

    Returns:
        None
    """
    ## Imporing torch to help with context issues
    import torch

    # Set seed values for various random number generators
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_network_interfaces() -> list[str]:
    """
    Gets a list of all valid network interfaces on a machine

    Returns:
        A list of all valid network interfaces on a machine
    """
    return list(psutil.net_if_addrs().keys())


def get_gpu_memory_info() -> dict[str, int]:
    """
    Get the total GPU memory for each Dask worker.
    Returns:
        dict: A dictionary mapping Dask worker addresses ('IP:PORT') to their
        respective GPU memory (in bytes).
    Example:
        {'192.168.0.100:9000': 3.2e+10, '192.168.0.101:9000': 3.2e+10}
    Note:
        If there is no active Dask client, an empty dictionary is returned.
    """
    client = get_current_client()
    if client is None:
        return {}
    return client.run(get_device_total_memory)
