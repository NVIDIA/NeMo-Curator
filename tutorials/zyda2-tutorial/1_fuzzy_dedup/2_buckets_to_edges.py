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

import os

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"

import logging
import time

import dask_cudf

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.fuzzy_dedup import BucketsToEdges
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    # Input
    lsh_base_output_path = os.path.join(DATA_BASE, "fuzzy/lsh")
    lsh_buckets_output_path = os.path.join(
        lsh_base_output_path, "data/_buckets.parquet"
    )

    # Output
    buckets_to_edges_out = os.path.join(DATA_BASE, "fuzzy/buckets_to_edges/data")

    t0 = time.time()

    ddf_bk = dask_cudf.read_parquet(
        lsh_buckets_output_path,
        split_row_groups=False,
    )

    buckets_to_edges = BucketsToEdges(
        cache_dir=buckets_to_edges_out,
        id_fields=["dataset_id", "doc_id"],
    )

    ddf_b2e = buckets_to_edges(DocumentDataset(ddf_bk))

    logging.info(f"Time taken for Buckets to Edges: {time.time() - t0} s")
