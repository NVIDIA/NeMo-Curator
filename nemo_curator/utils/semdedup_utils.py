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
import random
import shutil
import time
from typing import List, Literal, Optional, Tuple

import cudf
import cupy as cp
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from dask.distributed import progress

from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir

L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"


def normalize_embeddings_col_in_df(
    df: cudf.DataFrame, embedding_col: str
) -> cudf.DataFrame:
    tensor = torch.Tensor(get_array_from_df(df, embedding_col))
    normalized_tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
    df[embedding_col] = normalized_tensor.tolist()
    return df


def get_array_from_df(df: cudf.DataFrame, embedding_col: str) -> cp.ndarray:
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def pairwise_cosine_similarity(
    cluster_reps: torch.Tensor,
    device: Literal["cuda", "cpu"],
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Compute pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity
    """
    # Move to device
    cluster_reps = cluster_reps.to(device)
    # Compute pairwise cosine similarity
    pairwise_sim_matrix = torch.mm(cluster_reps, cluster_reps.T)
    del cluster_reps
    # Get upper triangular matrix
    assert pairwise_sim_matrix.shape[0] == pairwise_sim_matrix.shape[1]
    triu_sim_mat = torch.triu(pairwise_sim_matrix, diagonal=1)
    # Get max similarity and indices
    max_values_and_indices = torch.max(triu_sim_mat, dim=0)
    max_similarity = max_values_and_indices[0]
    max_indices = max_values_and_indices[1]
    return cp.asarray(max_similarity, dtype=cp.float32), cp.asarray(max_indices)


def pairwise_cosine_similarity_batched(
    cluster_reps: torch.Tensor,
    device: Literal["cuda", "cpu"],
    batch_size: int = 1024,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Computes pairwise cosine similarity between cluster items,
    then replace to diagonal with zeros to ignore self similarity.
    This function is useful for large clusters where the pairwise similarity matrix
    does not fit into memory.
    We use a batched approach to compute the pairwise similarity matrix in batches.
    Memory requirements are O(N*B) where N is the number of items in the cluster and B is the batch size
    instead of O(N^2) for the full matrix.
    """
    cluster_reps = cluster_reps.to(device)
    max_similarity = torch.zeros(cluster_reps.shape[0], dtype=torch.float32, device=device)
    max_indices = torch.zeros(cluster_reps.shape[0], dtype=torch.int64, device=device)
    for start_idx in range(0, cluster_reps.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, cluster_reps.shape[0])
        batch = cluster_reps[start_idx:end_idx]
        pairwise_sim_matrix = torch.mm(cluster_reps, batch.T)
        triu_sim_matrix = torch.triu(pairwise_sim_matrix, diagonal=1 - start_idx)
        del batch, pairwise_sim_matrix
        max_values_and_indices = torch.max(triu_sim_matrix, dim=0)
        max_similarity[start_idx:end_idx] = max_values_and_indices[0]
        max_indices[start_idx:end_idx] = max_values_and_indices[1]

    return cp.asarray(max_similarity), cp.asarray(max_indices)


def get_semantic_matches_per_cluster(
    cluster_id: int,
    emb_by_clust_dir: str,
    id_col: str,
    output_dir: str,
    embedding_col: str,
    which_to_keep: str,
    batched_cosine_similarity: int = 1024,
) -> None:
    cluster_df = cudf.read_parquet(
        os.path.join(emb_by_clust_dir, f"nearest_cent={cluster_id}"),
        columns=[embedding_col, id_col, COSINE_DIST_TO_CENT_COL],
    )
    output_df_file_path = os.path.join(output_dir, f"cluster_{cluster_id}.parquet")
    if len(cluster_df) == 1:
        cluster_df["indices"] = [0]
        cluster_df["id"] = cluster_df[id_col]
        cluster_df["max_id"] = cluster_df[id_col]
        cluster_df["cosine_sim_score"] = [0]
        cluster_df = cluster_df[["indices", "id", "max_id", "cosine_sim_score"]]
        cluster_df.to_parquet(output_df_file_path)
        return

    if which_to_keep == "hard":
        cluster_df = cluster_df.sort_values(
            by=[COSINE_DIST_TO_CENT_COL, id_col], ascending=False, ignore_index=True
        )
    elif which_to_keep == "easy":
        cluster_df = cluster_df.sort_values(
            by=[COSINE_DIST_TO_CENT_COL, id_col], ascending=True, ignore_index=True
        )
    elif which_to_keep == "random":
        cluster_df = cluster_df.sample(frac=1).reset_index(drop=True)

    cluster_embeddings = torch.as_tensor(
        get_array_from_df(cluster_df, embedding_col), device="cuda"
    )
    ids = cluster_df[id_col]
    assert cluster_embeddings.shape[0] == len(ids)

    if batched_cosine_similarity > 0:
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            cluster_embeddings, "cuda", batched_cosine_similarity
        )
    else:
        max_similarity, max_indices = pairwise_cosine_similarity(
            cluster_embeddings, "cuda"
        )

    max_indices_id = cluster_df[id_col].iloc[max_indices].values
    points_to_remove_df = cudf.DataFrame(
        {
            "indices": list(range(len(ids))),
            "id": ids,
            "max_id": max_indices_id,
            "cosine_sim_score": max_similarity,
        }
    )
    points_to_remove_df.to_parquet(output_df_file_path)


def get_num_records_from_npy(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "rb") as f:
        # Read the header of the npy file
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape[0]


def _get_empty_results_df(id_col, id_col_type):
    meta_df = pd.DataFrame(
        {
            id_col: np.empty(0, dtype="int64"),
            "dist": np.empty(0, dtype="float32"),
            "cluster": np.empty(0, dtype="int32"),
        }
    )
    meta_df[id_col] = meta_df[id_col].astype(id_col_type)
    return meta_df


def prune_single_cluster(
    cluster_id: int,
    id_col: str,
    id_col_type: str,
    emb_by_clust_dir: str,
    semdedup_pruning_tables_dir: str,
    eps: float,
) -> cudf.DataFrame:
    """
    Processes data for a single cluster, applying pruning based on specified epsilon.

    Args:
        cluster_id (int): The specific cluster ID to process.
        id_col (str): The name of the ID column.
        id_col_type (str): The data type of the ID column.
        emb_by_clust_dir (str): Path to where clustered embeddings are stored.
        semdedup_pruning_tables_dir (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.

    Returns:
        cudf.DataFrame: A DataFrame of the pruned cluster data
    """
    cluster_dir = os.path.join(emb_by_clust_dir, f"nearest_cent={cluster_id}")
    try:
        df_cluster = cudf.read_parquet(
            cluster_dir, columns=[id_col, COSINE_DIST_TO_CENT_COL]
        ).assign(cluster=cluster_id)
    except FileNotFoundError:
        return _get_empty_results_df(id_col, id_col_type)

    pruning_table_fname = os.path.join(
        semdedup_pruning_tables_dir, f"cluster_{cluster_id}.parquet"
    )
    # TODO should we add max_id to for the user
    pruning_table = cudf.read_parquet(
        pruning_table_fname, columns=["id","cosine_sim_score"]
    )
    if pruning_table.shape[0] == 1:
        return df_cluster
    pruning_table = pruning_table[pruning_table["cosine_sim_score"] > 1 - eps][["id"]]
    # TODO we can avoid this merge if we add more columns to the pruning table
    # However that might increase memory consumption at that stage, keeping it as is for now
    return df_cluster.merge(pruning_table.rename(columns={"id" : id_col}), on=id_col, how="inner")


def extract_pruned_data(
    id_col: str,
    id_col_type: str,
    emb_by_clust_dir: str,
    semdedup_pruning_tables_dir: str,
    eps: float,
    n_clusters: int,
    output_parquet_path: str,
    logger: Optional[logging.Logger] = None,
    profile_dir: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Extracts pruned data from sorted clusters and saves it to a CSV file.

    Args:
        id_col (str): The name of the ID column.
        id_col_type (str): The data type of the ID column.
        emb_by_clust_dir (str): Path to where clustered embeddings are stored.
        semdedup_pruning_tables_dir (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.
        n_clusters (int): Number of clusters.
        output_csv_path (str): Path to save the output CSV file.
        logger (Optional[logging.Logger]): Logger object or path to store logs, defaults to None.
        profile_dir (str): If specified directory to write dask profile. Default is None.

    Returns:
        Tuple[int, int, int]: Number of kept records, removed records, and total records.
    """

    t0 = time.time()

    with performance_report_if_with_ts_suffix(
        profile_dir,
        "extracting-pruned-from-clusters",
    ):
        results_df = dd.from_map(
            prune_single_cluster,
            range(n_clusters),
            id_col=id_col,
            id_col_type=id_col_type,
            emb_by_clust_dir=emb_by_clust_dir,
            semdedup_pruning_tables_dir=semdedup_pruning_tables_dir,
            eps=eps,
        )
        results_df.to_parquet(output_parquet_path, index=False, ignore_index=True)
    if logger:
        logger.info(
            f"Time taken for Extracting Pruned Data : {time.time() - t0} and output written at {output_parquet_path}"
        )


def extract_dedup_data(
    eps,
    n_clusters,
    id_col,
    id_col_type,
    emb_by_clust_dir: str,
    semdedup_pruning_tables_dir: str,
    output_summary_file,
    output_parquet_path,
    logger: logging.Logger,
    profile_dir: Optional[str] = None,
) -> dd.DataFrame:
    """
    Extracts deduplicated data based on provided parameters and logs the process.

    Args:

    """

    extract_pruned_data(
        id_col=id_col,
        id_col_type=id_col_type,
        emb_by_clust_dir=emb_by_clust_dir,
        semdedup_pruning_tables_dir=semdedup_pruning_tables_dir,
        eps=eps,
        n_clusters=n_clusters,
        output_parquet_path=output_parquet_path,
        logger=logger,
        profile_dir=profile_dir,
    )

    kept = len(dd.read_parquet(output_parquet_path))
    total = len(dd.read_parquet(emb_by_clust_dir))
    removed = total - kept

    logger.info(
        f"DONE saving {kept} out of {total}. Removed: {removed}. Epsilon: {eps:.4f}"
    )
    result_dict = {
        "eps": [eps],
        "kept": [kept],
        "removed": [removed],
        "total": [total],
    }
    df = pd.DataFrame(result_dict)
    df.to_csv(output_summary_file, index=False)

    fps = [
        os.path.join(output_parquet_path, file_name)
        for file_name in os.listdir(output_parquet_path)
    ]
    ids_to_keep_df = dd.from_map(cudf.read_parquet, fps)
    return ids_to_keep_df
