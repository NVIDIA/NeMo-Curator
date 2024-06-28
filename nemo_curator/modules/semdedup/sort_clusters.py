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
import pathlib
import pprint
import time
from datetime import datetime
from typing import List

import cudf
import dask.bag as db
import numpy as np
import torch

from nemo_curator.modules.semdedup.utils import get_logger, parse_arguments
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_client_args


def assign_and_sort_clusters(
    id_col: str,
    save_folder: str,
    sorted_clusters_file_loc: str,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = True,
    cluster_ids=range(5000),
    logger: logging.Logger = None,
):
    """
    Assigns data points to clusters and sorts each cluster's items based on their distance to the cluster centroid.

    Args:
        id_col (str): The column name representing the unique identifier for each data point.
        sim_metric (str): The similarity metric to use for clustering. Defaults to "cosine".
        keep_hard (bool): When True, sorts cluster items in descending order by similarity to the cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to True.
        save_folder (str): The location of the K-means centroids file. Defaults to an empty string.
        sorted_clusters_file_loc (str): The location to save the sorted clusters file. Defaults to an empty string.
        cluster_ids (list): The range of cluster IDs to sort. Defaults to range(5000).
        logger (logging.Logger): A logger object to log messages. Defaults to None.

    Returns:
        None
    """
    # Step 3: Sort each class/cluster
    logger.info("Ranking...")
    kmeans_centroids_file_loc = pathlib.Path(save_folder, "kmeans_centroids.npy")
    kmeans_centroids = np.load(kmeans_centroids_file_loc)
    nearest_cent_file_loc = pathlib.Path(save_folder, "embs_by_nearest_center")

    start_time = time.time()

    cluster_ids_bag = db.from_sequence(cluster_ids, npartitions=len(cluster_ids))
    cluster_ids_bag.map(
        lambda cluster_c: rank_within_cluster(
            id_col=id_col,
            nearest_cent_file_loc=nearest_cent_file_loc,
            centroids=kmeans_centroids,
            sim_metric=sim_metric,
            keep_hard=keep_hard,
            kmeans_with_cos_dist=kmeans_with_cos_dist,
            cluster_ids=[cluster_c],
            sorted_clusters_file_loc=sorted_clusters_file_loc,
        )
    ).compute()

    logger.info(f"Time for ranking: {(time.time() - start_time) / 60:.2f} mins")
    logger.info("DONE!")


def rank_within_cluster(
    id_col: str,
    nearest_cent_file_loc: str,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = False,
    cluster_ids: List[int] = range(50000),
    sorted_clusters_file_loc: str = "",
):
    """
    Sorts each cluster's items by their distance to the cluster centroid.

    Args:
        id_col (str): The column name representing the unique identifier for each data point.
        nearest_cent_file_loc (str): The location of the nearest center files.
        centroids (np.ndarray): The centroids for each cluster.
        sim_metric (str): The similarity metric used to compute distances. Should be one of ["cosine"]. Defaults to "cosine".
        keep_hard (bool): When True, sorts cluster items in descending order by similarity to the cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to False.
        cluster_ids (List[int]): The list of cluster IDs to process. Defaults to range(50000).
        sorted_clusters_file_loc (str): The location to save the sorted clusters.

    Returns:
        None
    """

    assert sim_metric in [
        "cosine",
    ], "sim_metric should be in ['cosine']"
    os.makedirs(sorted_clusters_file_loc, exist_ok=True)

    missing_files = 0
    logger.info(f"sorted_clusters_file_loc: {sorted_clusters_file_loc}")

    for cluster_c in cluster_ids:
        if os.path.exists(f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"):
            logger.info(f"Cluster {cluster_c} exits, skipping....")
            continue

        cluster_c_path = os.path.join(
            nearest_cent_file_loc, f"nearest_cent={cluster_c}"
        )
        if not os.path.exists(cluster_c_path):
            logger.info(f"{cluster_c_path} not found, skipping....")
            missing_files += 1
            continue

        cluster_df = cudf.read_parquet(
            cluster_c_path, columns=[id_col, "dist_to_cent", "embeddings"]
        )
        embeds = torch.as_tensor(
            cluster_df["embeddings"].list.leaves.values.reshape(
                cluster_df.shape[0], -1
            ),
            device="cuda",
        )
        cluster_df = cluster_df.to_pandas()

        assert kmeans_with_cos_dist is False

        if sim_metric == "cosine":
            cluster_c_centroid = torch.as_tensor(centroids[cluster_c], device="cuda")
            sim_to_cent = torch.nn.CosineSimilarity(dim=1)(embeds, cluster_c_centroid)
            sim_to_cent = sim_to_cent.cpu().numpy()
            cluster_dists_to_cent = (1 - sim_to_cent).tolist()
        elif sim_metric == "l2":
            # Used when Kmeans_with_cos_dist is True
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        example_id = list(cluster_df[id_col])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(example_id, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[2],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort

        sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        np.save(sorted_cluster_file_path, cluster_sorted)


if __name__ == "__main__":
    args = parse_arguments()
    # Initialize dask client
    client = get_client(**parse_client_args(args))

    save_loc = os.path.join(args.root, args.clustering["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    with open(pathlib.Path(save_loc, "sort_cluster_params.txt"), "w") as f:
        pprint.pprint(args, f)

    cluster_ids = list(range(args.clustering["num_clusters"]))
    logger = get_logger(
        file_name=f"{save_loc}/sort-cluster.log",
        level=logging.INFO,
        stdout=True,
    )

    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")

    kmeans_with_cos_dist = args.clustering["Kmeans_with_cos_dist"]
    assert kmeans_with_cos_dist is False
    which_to_keep = args.semdedup["which_to_keep"]
    keep_hard = which_to_keep == "hard"

    id_col = args.id_col["name"]
    assign_and_sort_clusters(
        id_col=id_col,
        sim_metric=args.semdedup["sim_metric"],
        keep_hard=keep_hard,
        kmeans_with_cos_dist=kmeans_with_cos_dist,
        save_folder=save_loc,
        sorted_clusters_file_loc=f"{save_loc}/sorted",
        cluster_ids=range(0, args.clustering["num_clusters"]),
        logger=logger,
    )

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")

    client.cancel(client.futures, force=True)
    client.close()
