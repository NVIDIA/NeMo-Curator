import logging
import os
import pathlib
import pickle
import random
from datetime import datetime

import cudf
import dask
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import get_logger

from nemo_curator.modules.semdedup.utils import parse_arguments
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_client_args


def _semdedup(cluster_reps, device):
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
    # M1 = torch.argmax(triu_sim_mat, dim=0).cpu().numpy().tolist()
    return M, M1


def get_cluster_reps(
    cluster_id: int, emb_by_clust_loc: str, id_col: str, sorted_ids: str
):
    cluster_i_path = os.path.join(emb_by_clust_loc, f"nearest_cent={cluster_id}")
    cluster_reps = cudf.read_parquet(
        cluster_i_path, columns=["embeddings", id_col]
    ).sort_values(by=id_col)
    num = cluster_reps.shape[0]

    df_ = pd.DataFrame(
        {"sorted_ids": sorted_ids, "inverse_sort": list(range(num))}
    ).sort_values(by="sorted_ids")
    cluster_reps["inverse_sort_id"] = df_["inverse_sort"].values
    cluster_reps = cluster_reps.sort_values(by="inverse_sort_id")

    cluster_reps = torch.Tensor(
        cluster_reps["embeddings"].list.leaves.values.reshape(len(cluster_reps), -1)
    )  # , device="cuda")
    return cluster_reps


def process_cluster(
    cluster_id, emb_by_clust_loc, id_col, id_col_type, eps_list, save_loc
):
    df_file_loc = os.path.join(save_loc, f"dataframes/cluster_{cluster_id}.pkl")
    if os.path.exists(df_file_loc):
        logging.info(f"{df_file_loc} exists. Continue")
        return

    ## -- load cluster i representations
    sorted_clusters_path = f"{save_loc}/sorted"
    sorted_file = os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
    if not os.path.exists(sorted_file):
        logging.info(f"{sorted_file} does not exist. Continue")
        return

    cluster_i = np.load(sorted_file)

    # 1) store cluster size
    cluster_size = cluster_i.shape[0]
    logging.info(f"{cluster_id}: cluster_size: {cluster_size}")

    if cluster_size == 1:
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = [0]
        for eps in eps_list:
            ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            points_to_remove_df[f"eps={eps}"] = [False]
        if save_loc != "":
            ## --save df
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)
        return

    ## -- By default, we keep hard examples from groups
    clutser_items_indices = list(range(cluster_size))

    ## -- OR: shuffle cluster to keep random example from each group
    which_to_keep = args.semdedup["which_to_keep"].lower()
    if which_to_keep == "random":
        random.shuffle(clutser_items_indices)
        cluster_i = cluster_i[clutser_items_indices]
    ## -- OR: reverse cluster to keep easy examples
    elif which_to_keep == "easy":
        clutser_items_indices = clutser_items_indices[::-1]
        cluster_i = cluster_i[clutser_items_indices]

    # can be adlr_id (nemo-curator data) or id (c4 data)
    text_ids = cluster_i[:, 0].astype(id_col_type)

    cluster_reps = get_cluster_reps(cluster_id, emb_by_clust_loc, id_col, text_ids)
    M, M1 = _semdedup(cluster_reps, "cuda")
    assert cluster_reps.shape[0] == len(text_ids)

    idx = [i for i in range(len(M1))]
    M1_id = [text_ids[m] for m in M1]

    points_to_remove_df = pd.DataFrame()
    points_to_remove_df["indices"] = clutser_items_indices
    points_to_remove_df["id"] = text_ids
    points_to_remove_df["max_id"] = M1_id
    points_to_remove_df["cosine_sim_score"] = M.numpy().tolist()

    for eps in eps_list:
        ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
        eps_points_to_remove = M > 1 - eps
        points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

    if save_loc != "":
        ## --save df
        os.makedirs(os.path.dirname(df_file_loc), exist_ok=True)
        with open(df_file_loc, "wb") as file:
            pickle.dump(points_to_remove_df, file)
    return


def semdedup(args):
    dt1 = datetime.now()
    logger.info(f"semdedup: start {dt1}")

    end_cluster = args.clustering["num_clusters"]
    root = args.root
    emb_pqt_loc = os.path.join(root, args.embeddings["output_data_dir"])

    eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
    eps_list2 = [0.1 + x * 0.005 for x in range(34)]
    eps_list = eps_list1 + eps_list2
    logger.info(f"emb_pqt_loc: {emb_pqt_loc}")

    id_col = args.id_col["name"]
    id_col_type = args.id_col["type"]
    save_loc = f'{root}/{args.clustering["save_loc"]}'
    emb_by_clust_loc = pathlib.Path(save_loc, "embs_by_nearest_center")

    tasks = []
    for cluster_id in tqdm(range(end_cluster)):
        tasks.append(
            dask.delayed(process_cluster)(
                cluster_id, emb_by_clust_loc, id_col, id_col_type, eps_list, save_loc
            )
        )

    dask.compute(*tasks)

    dt2 = datetime.now()
    logger.info(f"semdedup: end {dt2}, elapse: {(dt2 - dt1).total_seconds()/60} min")

    return


if __name__ == "__main__":

    args = parse_arguments()
    client = get_client(**parse_client_args(args))

    save_loc = os.path.join(args.root, args.clustering["save_loc"])

    os.makedirs(save_loc, exist_ok=True)
    os.makedirs(os.path.join(save_loc, "dataframes"), exist_ok=True)

    logger = get_logger(
        file_name=f"{save_loc}/semdedup.log",
        level=logging.INFO,
        stdout=True,
    )
    logger.info(args)
    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")

    semdedup(args)

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")
