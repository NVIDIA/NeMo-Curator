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


import cudf
import dask_cuda
import numpy as np
from dask import config
from packaging.version import Version

from nemo_curator._compat import query_planning_enabled

dask_cuda_version = Version(dask_cuda.__version__)
USE_EXCOMMS = (
    dask_cuda_version >= Version("23.10") and dask_cuda_version < Version("24.06")
) or dask_cuda_version >= Version("24.08")


def write_partitioned_file(df, output_path, partition_on, batch_id):
    if len(df) == 0:
        return cudf.Series([True])

    cudf.io.parquet.write_to_dataset(
        df,
        output_path,
        partition_cols=[partition_on],
        filename=f"batch_{batch_id}.parquet",
    )
    return cudf.Series([True])


def rearange_by_column_direct(
    df,
    col,
    npartitions,
    ignore_index,
    excomms_default=USE_EXCOMMS,
):
    # Execute a "direct" shuffle operation without staging
    if config.get("explicit-comms", excomms_default):
        from dask_cuda.explicit_comms.dataframe.shuffle import (
            shuffle as explicit_comms_shuffle,
        )

        # Use explicit comms unless the user has
        # disabled it with the dask config system,
        # or we are using an older version of dask-cuda
        return explicit_comms_shuffle(
            df,
            [col],
            npartitions=npartitions,
            ignore_index=ignore_index,
        )

    elif query_planning_enabled():
        from dask_expr._collection import new_collection
        from dask_expr._shuffle import RearrangeByColumn

        # Use the internal dask-expr API
        return new_collection(
            RearrangeByColumn(
                frame=df.expr,
                partitioning_index=col,
                npartitions_out=npartitions,
                ignore_index=ignore_index,
                method="tasks",
                # Prevent staged shuffling by setting max_branch
                # to the number of input partitions + 1
                options={"max_branch": npartitions + 1},
            )
        )

    else:
        from dask.dataframe.shuffle import rearrange_by_column

        return rearrange_by_column(
            df,
            col=col,
            shuffle_method="tasks",
            # Prevent staged shuffling by setting max_branch
            # to the number of input partitions + 1
            max_branch=npartitions + 1,
            npartitions=npartitions,
            ignore_index=ignore_index,
        )
