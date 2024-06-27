import argparse
import logging
import math
import os
import pathlib
import pdb
import pprint
import random
import time
from glob import glob
from typing import List, Tuple, Union

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from utils import get_logger


def assign_and_sort_clusters(
    data: np.ndarray,
    id_col: str,
    id_list: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = False,
    save_folder: str = "",
    sorted_clusters_file_loc: str = "",
    cluster_ids=range(5000),
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Assigns data points to clusters and sorts each cluster items based on distance to its centroid.

    Args:
        data (np.memmap): A memory-mapped array containing the data points.
        sim_metric (str): The similarity metric to use for clustering. Defaults to "cosine".
        keep_hard (bool): When True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to False.
        save_folder (str): The location of the K-means centroids file. Defaults to "".
        sorted_clusters_file_loc (str): The location to save the sorted clusters file. Defaults to "".
        logger (logging.Logger): A logger object to log messages. Defaults to None.
        cluster_ids (list): The range of cluster IDs to sort. Defaults to range(5000).

    Returns:
        pd.DataFrame: A DataFrame containing the sorted clusters.
    """
    assert sim_metric in [
        "l2",
        "cosine",
    ], f"Unsupported similarity metric '{sim_metric}'."
    assert not (
        kmeans_with_cos_dist and sim_metric == "l2"
    ), "Cannot use cosine distance with L2 similarity metric."

    # If Kmeans_with_cos_dist is True, set spherical=True. This is the spherical parameter of faiss kmeans clustering.
    spherical = kmeans_with_cos_dist

    # Step 3: Sort each class/cluster
    logger.info("Ranking...")
    kmeans_centroids_file_loc = pathlib.Path(save_folder, "kmeans_centroids.npy")
    kmeans_centroids = np.load(kmeans_centroids_file_loc)
    nearest_cent_file_loc = pathlib.Path(save_folder, "embs_by_nearest_center")

    start_time = time.time()

    sorted_clusters = rank_within_cluster(
        data,
        nearest_cent_file_loc,
        kmeans_centroids,
        sim_metric,
        keep_hard,
        spherical,
        cluster_ids,
        sorted_clusters_file_loc,
    )
    logger.info(f"Time for ranking: {(time.time() - start_time) / 60:.2f} mins")
    logger.info("DONE!")

    return sorted_clusters


def rank_within_cluster(
    data: np.ndarray,
    nearest_cent_file_loc: str,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    spherical: bool = False,
    cluster_ids: List[int] = range(50000),
    sorted_clusters_file_loc: str = "",
) -> List[List[Tuple[str, int, float, int]]]:
    """
    Sorts each cluster items by the distance to the cluster centroid.
    Cluster is represented as list of tuples. Each tuple has 4 values:
        example_path: unique path to the example/image/text doc, for imagenet it could be something like "n04235860_14959.JPEG",
        example_id_in_dataset: int between 0 and cluster_size-1
        dist_to_cent: cosine distance to cluster centroid
        cluster_id: cluster number (from 0 to number of clusters)

    Arguments:
    data -- the data for which the clusters were created (np.ndarray or np.memmap)
    dist_df -- DataFrame with the distances between the data points and the centroids, nearest centroid for each example, and path to each example.
    centroids -- np.ndarray with the centroids for each cluster.
    sim_metric -- the similarity metric used to compute distances, should be one of ["cosine", "l2"]
    keep_hard -- a boolean ehen True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
    spherical -- a boolean True means spherical was used for computing centroids (used for cosine similarity).
    cluster_ids -- a list of cluster ids to process. Each slurm job will process part of the clusters.
    sorted_clusters_file_loc -- the location to save the sorted clusters.

    Returns:
    A list of cluster representations, where each representation is a list of tuples with 4 values.
    -- exampel for a cluster (the list bellow is sorted by dist_to_cent in descending order)
        [
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
                                        .
                                        .
                                        .
                                                                    ]
    """

    assert sim_metric in [
        "cosine",
        "l2",
    ], "sim_metric should be one of ['cosine', 'l2']"
    os.makedirs(sorted_clusters_file_loc, exist_ok=True)

    print("====>", sorted_clusters_file_loc)
    sorted_clusters_list = []
    for cluster_c in tqdm(cluster_ids):
        if os.path.exists(f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"):
            print(f"Cluster {cluster_c} exits, skipping....")
            continue

        cluster_c_path = os.path.join(
            nearest_cent_file_loc, f"nearest_cent={cluster_c:0.1f}"
        )
        cluster_df = cudf.read_parquet(
            cluster_c_path, columns=[id_col, "nearest_cent", "dist_to_cent"]
        )
        # cluster_items = cluster_df.index.to_arrow().tolist()  ## -- ids of examples in cluster c
        cluster_df = cluster_df.to_pandas()

        # cluster_items = list(cluster_df.index)  ## -- ids of examples in cluster c
        cluster_items = list(cluster_df[id_col])
        if sim_metric == "cosine":
            if spherical:
                cluster_dists_to_cent = list(1 - cluster_df["dist_to_cent"])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(
                    torch.Tensor(data[cluster_items]), cluster_c_centroid
                )
                cluster_dists_to_cent = (1 - sim_to_cent).tolist()

        elif sim_metric == "l2":  # -- get l2 distance from "dist_to_cent" array
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        example_id = list(cluster_df[id_col])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(example_id, cluster_items, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[2],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort

        sorted_clusters_list.append(
            cluster_sorted
        )  # -- Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
        sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"

        np.save(sorted_cluster_file_path, cluster_sorted)
    return sorted_clusters_list


if __name__ == "__main__":
    config_file = "./configs_cf.yml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    save_loc = f'{params["root"]}/{params["clustering"]["save_loc"]}'
    os.makedirs(save_loc, exist_ok=True)

    with open(pathlib.Path(save_loc, "sort_cluster_params.txt"), "w") as f:
        pprint.pprint(params, f)

    cluster_ids = list(range(params["clustering"]["num_clusters"]))

    logger = get_logger(
        file_name=f"{save_loc}/sort-cluster-logs.log",
        level=logging.INFO,
        stdout=True,
    )

    emb_pqt_loc = f'{params["root"]}/{params["embeddings"]["emb_parquet_path"]}'
    df = pd.read_parquet(emb_pqt_loc)
    data = df["embeddings"]
    id_col = params["id_col"]["name"]
    id_list = df[id_col]

    assign_and_sort_clusters(
        data=data,
        id_col=id_col,
        id_list=id_list,
        sim_metric=params["semdedup"]["sim_metric"],
        keep_hard=params["semdedup"]["keep_hard"],
        kmeans_with_cos_dist=params["clustering"]["Kmeans_with_cos_dist"],
        save_folder=save_loc,
        sorted_clusters_file_loc=f"{save_loc}/sorted",
        cluster_ids=range(0, params["clustering"]["num_clusters"]),
        logger=logger,
    )
