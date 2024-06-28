import logging
import os
import pathlib
import random
from datetime import datetime
from typing import List, Tuple

import cudf
import dask.bag as db
import numpy as np
import pandas as pd
import torch

from nemo_curator.modules.semdedup.utils import get_logger, parse_arguments
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_client_args


def _semdedup(
    cluster_reps: torch.Tensor, device: str
) -> Tuple[torch.Tensor, List[int]]:
    # compute pairwise cos sim between cluster items,
    # then replace to diagonal with zeros to ignore self similarity
    cluster_reps.to(device)
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    del cluster_reps
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    M1 = torch.max(triu_sim_mat, dim=0)[1].cpu().numpy().tolist()
    return M, M1


def get_cluster_reps(
    cluster_id: int, emb_by_clust_loc: str, id_col: str, sorted_ids: np.ndarray
) -> torch.Tensor:
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

    cluster_reps = torch.as_tensor(
        cluster_reps["embeddings"].list.leaves.values.reshape(len(cluster_reps), -1),
        device="cuda",
    )
    return cluster_reps


def process_cluster(
    cluster_id: int,
    emb_by_clust_loc: str,
    id_col: str,
    id_col_type: str,
    eps_list: List[float],
    save_loc: str,
) -> None:
    assert save_loc is not None
    assert save_loc != ""

    df_file_loc = os.path.join(save_loc, f"dataframes/cluster_{cluster_id}.parquet")
    if os.path.exists(df_file_loc):
        logging.info(f"{df_file_loc} exists. Continue")
        return

    sorted_clusters_path = f"{save_loc}/sorted"
    sorted_file = os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
    if not os.path.exists(sorted_file):
        logging.info(f"{sorted_file} does not exist. Continue")
        return

    cluster_i = np.load(sorted_file)

    cluster_size = cluster_i.shape[0]
    logging.info(f"{cluster_id}: cluster_size: {cluster_size}")

    if cluster_size == 1:
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = [0]
        for eps in eps_list:
            points_to_remove_df[f"eps={eps}"] = [False]
        points_to_remove_df.to_parquet(df_file_loc)
        return

    clutser_items_indices = list(range(cluster_size))

    which_to_keep = args.semdedup["which_to_keep"].lower()
    if which_to_keep == "random":
        random.shuffle(clutser_items_indices)
        cluster_i = cluster_i[clutser_items_indices]
    elif which_to_keep == "easy":
        clutser_items_indices = clutser_items_indices[::-1]
        cluster_i = cluster_i[clutser_items_indices]

    text_ids = cluster_i[:, 0].astype(id_col_type)

    cluster_reps = get_cluster_reps(cluster_id, emb_by_clust_loc, id_col, text_ids)
    M, M1 = _semdedup(cluster_reps, "cuda")
    assert cluster_reps.shape[0] == len(text_ids)

    # TODO: Below is not used , ask @Fay if we can remove it
    idx = [i for i in range(len(M1))]
    M1_id = [text_ids[m] for m in M1]

    points_to_remove_df = cudf.DataFrame()
    points_to_remove_df["indices"] = clutser_items_indices
    points_to_remove_df["id"] = text_ids
    points_to_remove_df["max_id"] = M1_id
    points_to_remove_df["cosine_sim_score"] = M.numpy().tolist()

    for eps in eps_list:
        eps_points_to_remove = M > 1 - eps
        points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

    points_to_remove_df.to_parquet(df_file_loc)
    return


# TODO: Rename below function
def semdedup(args, logger) -> None:
    dt1 = datetime.now()
    logger.info(f"semdedup: start {dt1}")

    end_cluster = args.clustering["num_clusters"]
    root = args.root
    emb_pqt_loc = os.path.join(root, args.embeddings["save_loc"])

    eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
    eps_list2 = [0.1 + x * 0.005 for x in range(34)]
    eps_list = eps_list1 + eps_list2
    logger.info(f"emb_pqt_loc: {emb_pqt_loc}")

    id_col = args.id_col["name"]
    id_col_type = args.id_col["type"]
    save_loc = f'{root}/{args.clustering["save_loc"]}'
    emb_by_clust_loc = pathlib.Path(save_loc, "embs_by_nearest_center")

    tasks = db.from_sequence(list(range(end_cluster)), npartitions=end_cluster).map(
        lambda cluster_id: process_cluster(
            cluster_id=cluster_id,
            emb_by_clust_loc=emb_by_clust_loc,
            id_col=id_col,
            id_col_type=id_col_type,
            eps_list=eps_list,
            save_loc=save_loc,
        )
    )
    tasks.compute()
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

    # TODO: Rename below function
    semdedup(args, logger)

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")

    client.cancel(client.futures, force=True)
    client.close()
