# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import pandas as pd
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
from datetime import datetime



def _semdedup(cluster, cluster_reps, device):
    dt1 = datetime.now()
    logger.info(f"_embeddup: start {dt1}")
    
    ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
    cluster_reps.to(device)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the combinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    ## -- if the max sim between one example and any other example is > 1-eps, remove this example
    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    M1 = torch.argmax(triu_sim_mat, dim=0).cpu().numpy().tolist()

    dt2 = datetime.now()
    logger.info(f"_embeddup: end {dt2}, elapse: {(dt2 - dt1).total_seconds()/60} min")

    return M, M1

def semdedup(params):
    dt1 = datetime.now()
    logger.info(f"semdedup: start {dt1}")

    end_cluster = params["clustering"]["num_clusters"]
    root = params['root']
    idmappath = params['embeddings']['idmappath']
    emb_memory_loc = params['embeddings']['emb_memory_loc']
    dataset_size = get_dataset_size(root, idmappath)
    embs_memory_loc = f'{root}/{emb_memory_loc}'
    emb_size = params['embeddings']['emb_size']
    embs = np.memmap(embs_memory_loc, dtype='float32', mode="r", shape=(dataset_size, emb_size))

    save_loc = params['clustering']['save_loc']
    save_loc = f'{root}/{save_loc}'
    for cluster_id in tqdm(range(end_cluster)):

        df_file_loc = os.path.join(
            save_loc, f"dataframes/cluster_{cluster_id}.pkl"
        )

        if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
            logger.info(f"{df_file_loc} exists, moving on")
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
        logger.info(f"cluster_size: {cluster_size}")

        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            for eps in params['eps_list']:
                ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                points_to_remove_df[f"eps={eps}"] = [False]
            if save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)
            logger.info("DONE cluster_id ", cluster_id)
            continue

        ## -- By default, we keep hard examples from groups
        clutser_items_indices = list(range(cluster_size))
            
        ## -- OR: shuffle cluster to keep random example from each group
        which_to_keep = params['semdedup']['which_to_keep']
        if which_to_keep.lower() == "random":
            random.shuffle(clutser_items_indices)
            cluster_i = cluster_i[clutser_items_indices]
        ## -- OR: reverse cluster to keep easy examples
        if which_to_keep.lower() == "easy":
            clutser_items_indices = clutser_items_indices[::-1]
            cluster_i = cluster_i[clutser_items_indices]

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 1].astype("int32")
        cluster_reps = embs[cluster_ids]
        cluster_reps = torch.tensor(cluster_reps)
        cluster_adlr = cluster_i[:, 0].astype("str")

        M, M1 = _semdedup(cluster_i, cluster_reps, params['use_gpu'])
        idx = [i for i in range(len(M1))]
        M1_adlr = [cluster_adlr[m] for m in M1]
        M1_global = [cluster_ids[m] for m in M1]

        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = clutser_items_indices
        points_to_remove_df["global_id"] = cluster_ids
        points_to_remove_df["adlr_id"] = cluster_adlr
        points_to_remove_df["max_id"] = M1
        points_to_remove_df["max_global_id"] = M1_global
        points_to_remove_df["max_adlr"] = M1_adlr

        for eps in params['eps_list']:
            ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            eps_points_to_remove = M > 1 - eps
            points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

        if save_loc != "":
            ## --save df
            os.makedirs(os.path.dirname(df_file_loc), exist_ok = True)
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)

        #step_time.append_cluster(time.time() - step_st)
        logger.info(f"DONE cluster: {cluster_id}")

    dt2 = datetime.now()
    logger.info(f"semdedup: start {dt2}, elapse: {(dt2 - dt1).total_seconds()/60} min")
    return


if __name__ == "__main__":
    config_file = "./configs.yaml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    
    save_loc = params['root']/params['clustering']["save_loc"]
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
    
    
    semdedup(params)
 
