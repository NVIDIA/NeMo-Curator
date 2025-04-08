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

import os
from functools import wraps
from typing import Any, Callable, List, Literal, Optional, Union

import dask.dataframe as dd

from nemo_curator.utils.distributed_utils import read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.import_utils import gpu_only_import

dask_cudf = gpu_only_import("dask_cudf")


class DocumentDataset:
    """
    A collection of documents and document metadata.
    Internally it may be distributed across multiple nodes, and may be on GPUs.
    """

    def __init__(self, dataset_df: dd.DataFrame):
        if not hasattr(dataset_df, "npartitions"):
            raise RuntimeError(
                "Please use DocumentDataset.from_pandas or DocumentDataset.from_cudf "
                "to initialize your Pandas/cuDF DataFrame to a DocumentDataset."
            )
        self.df = dataset_df

    def __len__(self) -> int:
        return len(self.df)

    # `def persist(self) -> Self` requires Python 3.11 or higher
    def persist(self) -> "DocumentDataset":
        return DocumentDataset(self.df.persist())

    @wraps(dd.DataFrame.repartition)
    def repartition(self, *args, **kwargs) -> "DocumentDataset":
        return self.__class__(self.df.repartition(*args, **kwargs))

    def head(self, n: int = 5) -> Any:
        return self.df.head(n)

    @classmethod
    def read_json(
        cls,
        input_files: Union[str, List[str]],
        backend: Literal["pandas", "cudf"] = "pandas",
        files_per_partition: Optional[int] = None,
        blocksize: Optional[str] = "1gb",
        add_filename: Union[bool, str] = False,
        input_meta: Optional[Union[str, dict]] = None,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read JSONL or JSONL file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            input_meta: A dictionary or a string formatted as a dictionary, which outlines
                the field names and their respective data types within the JSONL input file.
            columns: If not None, only these columns will be read from the file.

        """
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="jsonl",
                backend=backend,
                add_filename=add_filename,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                input_meta=input_meta,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_parquet(
        cls,
        input_files: Union[str, List[str]],
        backend: Literal["pandas", "cudf"] = "pandas",
        files_per_partition: Optional[int] = None,
        blocksize: Optional[str] = "1gb",
        add_filename: Union[bool, str] = False,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read Parquet file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be read from the file.
                There is a significant performance gain when specifying columns for Parquet files.

        """
        return cls(
            _read_json_or_parquet(
                input_files=input_files,
                file_type="parquet",
                backend=backend,
                add_filename=add_filename,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_pickle(
        cls,
        input_files: Union[str, List[str]],
        backend: Literal["pandas", "cudf"] = "pandas",
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read Pickle file(s).

        Args:
            input_files: The path of the input file(s).
            backend: The backend to use for reading the data.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be read from the file.

        """
        return cls(
            read_data(
                input_files=input_files,
                file_type="pickle",
                backend=backend,
                columns=columns,
                **kwargs,
            )
        )

    @classmethod
    def read_custom(
        cls,
        input_files: Union[str, List[str]],
        file_type: str,
        read_func_single_partition: Callable[
            [List[str], str, bool, Union[str, dict], dict],
            Union["cudf.DataFrame", "pd.DataFrame"],
        ],
        files_per_partition: Optional[int] = None,
        backend: Optional[Literal["pandas", "cudf"]] = None,
        add_filename: Union[bool, str] = False,
        columns: Optional[List[str]] = None,
        input_meta: Union[str, dict] = None,
        **kwargs,
    ) -> "DocumentDataset":
        """
        Read custom data from a file or directory based on a custom read function.


        Args:
            input_files: The path of the input file(s).
                If input_file is a string that ends with the file_type, we consider it as a single file.
                If input_file is a string that does not end with the file_type, we consider it as a directory
                and read all files under the directory.
                If input_file is a list of strings, we assume each string is a file path.
            file_type: The type of the file to read.
            read_func_single_partition: A function that reads a single file or a list of files in an single dask partition.
                The function should take the following arguments:
                - files: A list of file paths.
                - file_type: The type of the file to read (in case you want to handle different file types differently).
                - backend: Read below
                - add_filename: Read below
                - columns: Read below
                - input_meta: Read below
            backend: The backend to use for reading the data, in case you want to handle pd.DataFrame or cudf.DataFrame.
            files_per_partition: The number of files to read per partition.
            add_filename: Whether to add a filename column to the DataFrame.
                If True, a new column is added to the DataFrame called `file_name`.
                If str, sets new column name. Default is False.
            columns: If not None, only these columns will be returned from the output of the read_func_single_partition function.
            input_meta: A dictionary or a string formatted as a dictionary, which outlines
                the field names and their respective data types within the JSONL input file.
        """
        if isinstance(input_files, str):
            if input_files.endswith(file_type):
                files = [input_files]
            else:
                files = get_all_files_paths_under(
                    root=input_files,
                    recurse_subdirectories=False,
                    keep_extensions=[file_type],
                )
        elif isinstance(input_files, list):
            files = input_files
        else:
            raise TypeError("input_files must be a string or list")
        return cls(
            read_data(
                input_files=files,
                backend=backend,
                files_per_partition=files_per_partition,
                blocksize=None,
                add_filename=add_filename,
                columns=columns,
                input_meta=input_meta,
                read_func_single_partition=read_func_single_partition,
                **kwargs,
            )
        )

    def to_json(
        self,
        output_path: str,
        write_to_filename: Union[bool, str] = False,
        keep_filename_column: bool = False,
        partition_on: Optional[str] = None,
        compression: Optional[str] = None,
    ):
        """
        Writes the dataset to the specified path in JSONL format.

        If `write_to_filename` is True, the DataFrame is expected to have a column
        that specifies the filename for each document. This column can be named
        `file_name` by default, or a custom name if `write_to_filename` is a string.

        Args:
            output_path (str): The directory or file path where the dataset will be written.
            write_to_filename (Union[bool, str]): Determines how filenames are handled.
                - If True, uses the `file_name` column in the DataFrame to determine filenames.
                - If a string, uses that string as the column name for filenames.
                - If False, writes all data to the specified `output_path`.
            keep_filename_column (bool): If True, retains the filename column in the output.
                If False, the filename column is dropped from the output.
            partition_on (Optional[str]): The column name used to partition the data.
                If specified, data is partitioned based on unique values in this column,
                with each partition written to a separate directory.
            compression (Optional[str]): The compression to use for the output file.
                If specified, the output file will be compressed using the specified compression.
                Supported compression types are "gzip" or None.
        For more details, refer to the `write_to_disk` function in
        `nemo_curator.utils.distributed_utils`.
        """
        write_to_disk(
            df=self.df,
            output_path=output_path,
            write_to_filename=write_to_filename,
            keep_filename_column=keep_filename_column,
            partition_on=partition_on,
            output_type="jsonl",
            compression=compression,
        )

    def to_parquet(
        self,
        output_path: str,
        write_to_filename: Union[bool, str] = False,
        keep_filename_column: bool = False,
        partition_on: Optional[str] = None,
    ):
        """
        Writes the dataset to the specified path in Parquet format.

        If `write_to_filename` is True, the DataFrame is expected to have a column
        that specifies the filename for each document. This column can be named
        `file_name` by default, or a custom name if `write_to_filename` is a string.

        Args:
            output_path (str): The directory or file path where the dataset will be written.
            write_to_filename (Union[bool, str]): Determines how filenames are handled.
                - If True, uses the `file_name` column in the DataFrame to determine filenames.
                - If a string, uses that string as the column name for filenames.
                - If False, writes all data to the specified `output_path`.
            keep_filename_column (bool): If True, retains the filename column in the output.
                If False, the filename column is dropped from the output.
            partition_on (Optional[str]): The column name used to partition the data.
                If specified, data is partitioned based on unique values in this column,
                with each partition written to a separate directory.

        For more details, refer to the `write_to_disk` function in
        `nemo_curator.utils.distributed_utils`.
        """
        write_to_disk(
            df=self.df,
            output_path=output_path,
            write_to_filename=write_to_filename,
            keep_filename_column=keep_filename_column,
            partition_on=partition_on,
            output_type="parquet",
        )

    def to_pickle(
        self,
        output_path: str,
        write_to_filename: Union[bool, str] = False,
    ):
        raise NotImplementedError("DocumentDataset does not support to_pickle yet")

    @classmethod
    def from_pandas(
        cls,
        data,
        npartitions: Optional[int] = 1,
        chunksize: Optional[int] = None,
        sort: Optional[bool] = True,
        name: Optional[str] = None,
    ):
        """
        Creates a document dataset from a Pandas DataFrame.
        For more information on the arguments see Dask's from_pandas documentation
        https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html

        Args:
            data: A Pandas DataFrame
        Returns:
            A DocumentDataset with a Pandas backend (on the CPU).
        """
        return cls(
            dd.from_pandas(
                data=data,
                npartitions=npartitions,
                chunksize=chunksize,
                sort=sort,
            )
        )

    def to_pandas(self):
        """
        Creates a Pandas DataFrame from a DocumentDataset

        Returns:
            A Pandas DataFrame (on the CPU)
        """
        return self.df.to_backend("pandas").compute()

    @classmethod
    def from_cudf(
        cls,
        data,
        npartitions: Optional[int] = 1,
        chunksize: Optional[int] = None,
        sort: Optional[bool] = True,
        name: Optional[str] = None,
    ):
        """
        Creates a document dataset from a cuDF DataFrame.
        For more information on the arguments see Dask-cuDF's from_cudf documentation
        https://docs.rapids.ai/api/dask-cudf/legacy/api/

        Args:
            data: A cuDF DataFrame
        Returns:
            A DocumentDataset with a cuDF backend (on the GPU).
        """
        return cls(
            dask_cudf.from_cudf(
                data=data,
                npartitions=npartitions,
                chunksize=chunksize,
                sort=sort,
            )
        )

    def to_cudf(self):
        """
        Creates a cuDF DataFrame from a DocumentDataset

        Returns:
            A cuDF DataFrame (on the GPU)
        """
        return self.df.to_backend("cudf").compute()


def _read_json_or_parquet(
    input_files: Union[str, List[str]],
    file_type: str,
    backend: Literal["cudf", "pandas"],
    add_filename: Union[bool, str] = False,
    files_per_partition: Optional[int] = None,
    blocksize: Optional[str] = None,
    input_meta: Union[str, dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
):
    """
    `input_files` may be a list or a string type.
    If `input_files` is a list, it may be a list of JSONL or Parquet files,
    e.g., `input_files = ["file1.jsonl", "file2.jsonl", "file3.jsonl"]`,
    or a list of directories containing JSONL or Parquet files,
    e.g., `input_files = ["dir1", "dir2", "dir3"]`,
    where each of `dir1`, `dir2`, and `dir3` contain all JSONL or Parquet files.
    If `input_files` is a string, it may be a single JSONL or Parquet file,
    such as `input_files = "my_file.jsonl"`,
    or it may also be a single directory containing JSONL or Parquet files,
    such as `input_files = "my_directory"`.

    See nemo_curator.utils.distributed_utils.read_data docstring for other parameters.

    Returns a DataFrame to be used in initializing a DocumentDataset.

    """
    file_ext = "." + file_type

    if isinstance(input_files, list):
        # List of files
        if all(os.path.isfile(f) for f in input_files):
            raw_data = read_data(
                input_files,
                file_type=file_type,
                backend=backend,
                files_per_partition=files_per_partition,
                blocksize=blocksize,
                add_filename=add_filename,
                input_meta=input_meta,
                columns=columns,
                **kwargs,
            )

        # List of directories
        else:
            dfs = []

            for data_path in input_files:
                files = get_all_files_paths_under(
                    root=data_path, recurse_subdirectories=False
                )
                df = read_data(
                    files,
                    file_type=file_type,
                    backend=backend,
                    files_per_partition=files_per_partition,
                    blocksize=blocksize,
                    add_filename=add_filename,
                    input_meta=input_meta,
                    columns=columns,
                    **kwargs,
                )
                dfs.append(df)

            raw_data = dd.concat(dfs, ignore_unknown_divisions=True)

    elif isinstance(input_files, str):
        # Single file
        if input_files.endswith(file_ext):
            files = [input_files]

        # Directory of jsonl or parquet files
        else:
            files = get_all_files_paths_under(
                root=input_files, recurse_subdirectories=False
            )

        raw_data = read_data(
            input_files=files,
            file_type=file_type,
            backend=backend,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
            add_filename=add_filename,
            input_meta=input_meta,
            columns=columns,
            **kwargs,
        )

    else:
        raise TypeError("File input must be a string or list.")

    return raw_data
