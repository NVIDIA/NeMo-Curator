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
import json
import os
import time

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"

import logging

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo quality classifier.")
    parser.add_argument(
        "--dupes-path",
        type=str,
        required=True,
        help="Path to the folder with dupes indices",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the folder with input dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to where write the result"
    )
    parser.add_argument(
        "--n-workers", type=int, default=64, help="Number of CPU Dask workers"
    )
    args = parser.parse_args()

    t0 = time.time()
    cluster = LocalCluster(
        n_workers=args.n_workers, threads_per_worker=2, processes=True
    )
    client = Client(cluster)
    logging.info(f"Dask client: {client}")
    logging.info(f"Dashboard link: {client.dashboard_link}")

    dupes_files = sorted([x for x in os.listdir(args.dupes_path) if ".jsonl" in x])
    ind_2_fullpath = {}
    for file in dupes_files:
        ind = int(file.split(".")[0])
        ind_2_fullpath[ind] = os.path.join(args.dupes_path, file)
    logging.info(f"Found {len(dupes_files)} files with dupes")

    ddf = dd.read_parquet(args.input, split_row_groups=False)

    def drop_dupes(partition, partition_info=None):
        ind = partition_info["number"]

        dupes_file_path = ind_2_fullpath.get(ind, None)
        if not dupes_file_path:
            return partition

        with open(dupes_file_path, "r") as f:
            dupe_inds = json.loads(f.read())["rows"]
            partition_deduped = partition.drop(index=dupe_inds)
            return partition_deduped

    meta = ddf.dtypes.to_dict()
    ddf_deduped = ddf.map_partitions(drop_dupes, meta=meta)
    logging.info(f"Removing dupes...")
    ddf_deduped.to_parquet(args.output, write_index=False, overwrite=True)
    logging.info(f"Done in {time.time() - t0:.2f}sec")

    client.cluster.close()
    client.shutdown()
