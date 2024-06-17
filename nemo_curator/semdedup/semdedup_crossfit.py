# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import pandas as pd
import submitit
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
from utils import get_logger, get_dataset_size
import yaml
import pathlib
import logging
import argparse
from typing import List, Tuple
import cudf
from glob import glob
from dask.distributed import LocalCluster
from dask.distributed import Client
from datetime import datetime
from dask_cuda import LocalCUDACluster
import dask_cudf
import cupy as cp

def _semdedup(cluster, cluster_reps, device):
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    cluster_reps.to(device)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the combinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    ## -- if the max sim between one example and any other example is > 1-eps, remove this example
    # torch.max returns values and indices. Indices is argmax
    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    M1 = torch.max(triu_sim_mat, dim=0)[1].cpu().numpy().tolist()
    #M1 = torch.argmax(triu_sim_mat, dim=0).cpu().numpy().tolist()
    return M, M1


def get_embedding_ar(df):
    return df['embeddings'].list.leaves.values.reshape(len(df),-1)

def find_cluster_reps(emb_pqt_loc, id_col, id_list, emb_size):
    ddf = dask_cudf.read_parquet(emb_pqt_loc, columns=["embeddings", id_col])
    ddf1 = ddf[ddf[id_col].isin(id_list)]
    ddf1 = ddf1.sort_values(by=[id_col])
    num_workers = 1
    ddf1 = ddf1.repartition(npartitions=num_workers)
    cupy_darr = ddf1.map_partitions(get_embedding_ar, meta=cp.ndarray([1, emb_size]))
    cupy_darr = cupy_darr.compute() 
    assert cupy_darr.shape[0] == len(id_list)
    return cupy_darr


def semdedup(params):
    dt1 = datetime.now()
    logger.info(f"semdedup: start {dt1}")

    end_cluster = params["clustering"]["num_clusters"]
    root = params["root"]
    emb_pqt_loc = f'{root}/{params["embeddings"]["emb_parquet_path"]}'
    logger.info(f"emb_pqt_loc: {emb_pqt_loc}")

    emb_size = params["embeddings"]["emb_size"]
    id_col = params["id_col"]["name"]
    id_col_type = params["id_col"]["type"]
    save_loc = f'{root}/{params["clustering"]["save_loc"]}'
    result_dir = f'{save_loc}/dataframes'

    for cluster_id in tqdm(range(end_cluster)):

        df_file_loc = os.path.join(
            save_loc, f"dataframes/cluster_{cluster_id}.pkl"
        )

        if os.path.exists(df_file_loc):
            logger.info(f"{df_file_loc} exists. Continue")
            continue

        ## -- load cluster i representations
        sorted_clusters_path = f"{save_loc}/sorted"
        cluster_i = np.load(
            os.path.join(
                sorted_clusters_path, f"cluster_{cluster_id}.npy"
            )
        )
        # 1) store cluster size
        cluster_size = cluster_i.shape[0]
        logger.info(f"{cluster_id}: cluster_size: {cluster_size}")

        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in params["eps_list"]:
                ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                points_to_remove_df[f"eps={eps}"] = [False]
            if save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)
            continue

        ## -- By default, we keep hard examples from groups
        clutser_items_indices = list(range(cluster_size))
            
        ## -- OR: shuffle cluster to keep random example from each group
        which_to_keep = params["semdedup"]["which_to_keep"]
        if which_to_keep.lower() == "random":
            random.shuffle(clutser_items_indices)
            cluster_i = cluster_i[clutser_items_indices]
        ## -- OR: reverse cluster to keep easy examples
        if which_to_keep == "easy":
            clutser_items_indices = clutser_items_indices[::-1]
            cluster_i = cluster_i[clutser_items_indices]

        ## -- indices for cluster items in the dataset
        cluster_global_ids = cluster_i[:, 1].astype("int32")
        cluster_global_ids.sort()
        
        #cluster_reps = embs[cluster_ids]
        #cluster_reps = torch.tensor(cluster_reps)

        # can be adlr_id (nemo-curator data) or id (c4 data)

        cluster_ids = cluster_i[:, 0].astype(id_col_type)
        cluster_ids.sort()

        cluster_reps = find_cluster_reps(emb_pqt_loc, id_col, cluster_ids, emb_size)
        cluster_reps = torch.tensor(cluster_reps)


        M, M1 = _semdedup(cluster_i, cluster_reps, params["use_gpu"])

        idx = [i for i in range(len(M1))]
        M1_id = [cluster_ids[m] for m in M1]
        M1_global = [cluster_global_ids[m] for m in M1]
 
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = clutser_items_indices
        points_to_remove_df["global_id"] = cluster_global_ids 
        points_to_remove_df["id"] = cluster_ids
        points_to_remove_df["max_id"] = M1
        points_to_remove_df["max_global_id"] = M1_global
        points_to_remove_df["max_id"] = M1_id
        points_to_remove_df["cosine_sim_score"] = M.numpy().tolist()

        for eps in params["eps_list"]:
            ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            eps_points_to_remove = M > 1 - eps
            points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

        if save_loc != "":
            ## --save df
            os.makedirs(os.path.dirname(df_file_loc), exist_ok = True)
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)
        
    dt2 = datetime.now()
    logger.info(f"semdedup: start {dt2}, elapse: {(dt2 - dt1).total_seconds()/60} min")

    return

if __name__ == "__main__":
    config_file = "./configs_cf.yml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    save_loc = f"{params['root']}/{params['clustering']['save_loc']}"
    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(f'{save_loc}/dataframes', exist_ok=True)

    logger = get_logger(
            file_name=f"{save_loc}/semdedup-logs.log",
            level=logging.INFO,
            stdout=True,
        )



    params['eps_list'] = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
    use_gpu = torch.cuda.is_available()
   
    params['use_gpu'] = "cuda" if use_gpu else "cpu"

    logger.info(params)
    dt1 = datetime.now()
    logger.info(f'Start: {dt1}')
    
    cluster = LocalCUDACluster() if use_gpu else LocalCluster()
    client = Client(cluster)
    semdedup(params)

    dt2 = datetime.now()
    logger.info(f'End: {dt2}')
    elapse = (dt2 - dt1).total_seconds()/60
    logger.info(f'elapse: {elapse}')












 
