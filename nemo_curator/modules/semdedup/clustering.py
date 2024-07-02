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

import logging
import os
from datetime import datetime

import dask_cudf

from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup import ClusteringModel
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_client_args, parse_semdedup_args

if __name__ == "__main__":
    # Configure command line arguments
    semdedup_config = SemDedupConfig.from_yaml("configs/config.yaml")
    parser = parse_semdedup_args(add_input_args=False)
    args = parser.parse_args()

    save_folder = os.path.join(
        semdedup_config.cache_dir, semdedup_config.clustering["save_loc"]
    )
    os.makedirs(save_folder, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(save_folder, "compute_centroids.log")

    logger = create_logger(
        rank=0,
        log_file=log_file,
        log_level=logging.INFO,
        name="logger-compute-centroids",
        stdout=True,
    )

    client = get_client(**parse_client_args(args))
    dt1 = datetime.now()
    print("Start time:", dt1)

    embedding_fp = os.path.join(
        semdedup_config.cache_dir, semdedup_config.embeddings["save_loc"]
    )

    clustering_output_dir = os.path.join(
        semdedup_config.cache_dir, semdedup_config.clustering["save_loc"]
    )
    embedding_df = dask_cudf.read_parquet(embedding_fp, blocksize="4GB")

    clustering_model = ClusteringModel(
        max_iter=semdedup_config.clustering["max_iter"],
        n_clusters=semdedup_config.clustering["n_clusters"],
        clustering_output_dir=clustering_output_dir,
        logger=logger,
    )

    clustered_embeddings = clustering_model(embedding_df)
    clustered_embeddings.head(10)
    dt2 = datetime.now()
    elapse = dt2 - dt1
    print("End time:", dt2)
    print("elapse:", elapse)

    client.cancel(client.futures, force=True)
    client.close()
