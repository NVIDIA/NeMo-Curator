# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pathlib
import pprint
import time
from datetime import datetime
from typing import Optional, Union

import cupy as cp
import dask
import dask.array as da
import dask.dataframe as dd
import dask_cudf
import numpy as np
import pandas as pd
import torch
import yaml
from cuml.dask.cluster import KMeans
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from utils import get_logger


def get_embedding_ar(df):
    return df["embeddings"].list.leaves.values.reshape(len(df), -1)


def compute_centroids(client, params):
    ## -- Load clustering parameters
    emb_pqt_loc = f'{params["root"]}/{params["embeddings"]["emb_parquet_path"]}'
    emb_size = params["embeddings"]["emb_size"]
    niter = params["clustering"]["niter"]
    ncentroids = params["clustering"]["num_clusters"]

    save_folder = f'{params["root"]}/{params["clustering"]["save_loc"]}'
    os.makedirs(save_folder, exist_ok=True)

    ddf = dask_cudf.read_parquet(
        emb_pqt_loc, columns=["embeddings", params["id_col"]["name"]]
    )
    num_workers = len(client.scheduler_info()["workers"])
    # Persist ddf to save IO costs
    # Probably should enable spilling
    ddf = ddf.repartition(npartitions=num_workers).persist()
    cupy_darr = ddf.map_partitions(get_embedding_ar, meta=cp.ndarray([1, emb_size]))
    cupy_darr.compute_chunk_sizes()
    kmeans = KMeans(n_clusters=ncentroids, init_max_iter=niter, oversampling_factor=10)

    dist_to_cents = kmeans.fit_transform(cupy_darr)
    dist_to_cents = dist_to_cents.min(axis=1)

    nearest_cents = kmeans.predict(cupy_darr)
    ddf["nearest_cent"] = nearest_cents
    centroids = kmeans.cluster_centers_
    ddf["dist_to_cent"] = dist_to_cents

    return centroids, ddf


if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--confg-file",
        type=str,
        default="./configs_cf.yml",
        help=".yaml config file path",
    )

    args = parser.parse_args()
    confg_file = args.confg_file

    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    save_folder = f'{params["root"]}/{params["clustering"]["save_loc"]}'
    os.makedirs(save_folder, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(save_folder, "compute_centroids.log")
    logger = get_logger(
        file_name=log_file,
        level=logging.INFO,
        stdout=True,
    )

    with open(pathlib.Path(save_folder, "clustering_params.txt"), "w") as fout:
        pprint.pprint(params, fout)

    kmeans_file_loc = pathlib.Path(save_folder, "kmeans_centroids.npy")
    if not os.path.exists(kmeans_file_loc):
        dt1 = datetime.now()
        print("Start time:", dt1)

        cluster = LocalCUDACluster()
        client = Client(cluster)
        centroids, ddf = compute_centroids(client, params)

        # ddf.to_parquet(f"{save_folder}/added_nearest_center.parquet", index=False)
        ddf.to_parquet(
            f"{save_folder}/embs_by_nearest_center/",
            index=False,
            partition_on="nearest_cent",
        )
        kmeans_centroids_file = pathlib.Path(save_folder, "kmeans_centroids.npy")
        np.save(kmeans_centroids_file, centroids)

        dt2 = datetime.now()
        elapse = dt2 - dt1
        print("End time:", dt2)
        print("elapse:", elapse)
