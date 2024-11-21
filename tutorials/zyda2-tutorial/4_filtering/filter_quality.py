# Copyright (c) 2024, NVIDIA CORPORATION.
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

import argparse
import os
import time

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"

import logging

import dask.dataframe as dd
import pyarrow as pa
from dask.distributed import Client, LocalCluster

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter by quality")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the folder with input dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to where write the result"
    )
    parser.add_argument(
        "--n-workers", type=int, default=64, help="Number of CPU Dask workers"
    )
    parser.add_argument(
        "--quality_pred",
        type=str,
        required=True,
        choices={"High", "Medium", "Low"},
        help="Quality for filtering",
    )
    args = parser.parse_args()

    t0 = time.time()
    cluster = LocalCluster(
        n_workers=args.n_workers, threads_per_worker=2, processes=True
    )
    client = Client(cluster)

    ddf = dd.read_parquet(args.input, split_row_groups=False)
    ddf_filtered = ddf[ddf["quality_pred"] == args.quality_pred].repartition(
        partition_size="512MB"
    )
    ddf_filtered.to_parquet(
        args.output,
        write_index=False,
        overwrite=True,
        schema={"quality_prob": pa.list_(pa.float32())},
    )
    ddf_filtered = dd.read_parquet(args.output)
    l_after = len(ddf_filtered)
    logging.info(f"Done in {time.time() - t0:.2f} sec")

    client.cluster.close()
    client.shutdown()
