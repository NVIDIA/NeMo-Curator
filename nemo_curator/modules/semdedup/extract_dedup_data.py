import logging
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig


def extract_pruned_data(
    id_col: str,
    id_type: str,
    sorted_clusters_path: str,
    semdedup_pruning_tables_path: str,
    eps: float,
    num_clusters: int,
    output_csv_path: str,
) -> Tuple[int, int, int]:
    """
    Extracts pruned data from sorted clusters and saves it to a CSV file.

    Args:
        id_col (str): The name of the ID column.
        id_type (str): The data type of the ID column.
        sorted_clusters_path (str): Path to the sorted clusters directory.
        semdedup_pruning_tables_path (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.
        num_clusters (int): Number of clusters.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        Tuple[int, int, int]: Number of kept records, removed records, and total records.
    """
    dedup_clusters = []
    total = 0

    for cluster_id in tqdm(range(num_clusters)):

        sorted_fname = os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
        if not os.path.exists(sorted_fname):
            logger.info(f"{sorted_fname} not exist. Continue.")
            continue
        cluster_i = np.load(sorted_fname)

        df_cluster_i = pd.DataFrame(
            {
                id_col: cluster_i[:, 0],
                "dist": cluster_i[:, 1],
                "cluster": cluster_i[:, 2],
            }
        )
        df_cluster_i[id_col] = df_cluster_i[id_col].astype(id_type)
        df_cluster_i.dist = df_cluster_i.dist.astype("float32")
        df_cluster_i.cluster = df_cluster_i.cluster.astype("int32")
        total += df_cluster_i.shape[0]
        cluster_df_fname = os.path.join(
            semdedup_pruning_tables_path, f"cluster_{cluster_id}.parquet"
        )
        semdedup_pruning_tables = pd.read_parquet(cluster_df_fname)

        if semdedup_pruning_tables.shape[0] == 1:
            logger.info(
                f"""cluster_id: {cluster_id},
                    semdedup_pruning_tables.shape: {semdedup_pruning_tables.shape},
                    df_cluster_i.shape: {df_cluster_i.shape}"""
            )
            continue

        items_to_keep = semdedup_pruning_tables[
            semdedup_pruning_tables[f"eps={eps}"] == False
        ]["id"].tolist()

        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]

        dedup_cluster = df_cluster_i[df_cluster_i[id_col].isin(items_to_keep)]
        dedup_clusters.append(dedup_cluster)

    result = pd.concat(dedup_clusters)
    result.to_csv(output_csv_path, index=False)
    num_removed = total - result.shape[0]

    logger.info(f"DONE saving {result.shape[0]} out of {total}. Removed: {num_removed}")
    return result.shape[0], num_removed, total


def extract_dedup_data(semdedup_config: SemDedupConfig, logger: logging.Logger) -> None:
    """
    Extracts deduplicated data based on provided parameters and logs the process.

    Args:
        semdedup_config: Configuration object for SemDedup.
        logger (logging.Logger): Logger for logging the process.
    """

    if semdedup_config.extract_dedup["use_eps_from_yml"]:
        eps = semdedup_config.extract_dedup["eps"]
        eps_list = [float(x) for x in eps.split(" ")]
    else:
        eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
        eps_list2 = [0.1 + x * 0.005 for x in range(34)]
        eps_list = eps_list1 + eps_list2
    kept_list = []
    removed_list = []
    total_list = []
    id_col = semdedup_config.id_col["name"]
    id_type = semdedup_config.id_col["type"]

    for eps in eps_list:
        output_csv_path = f"{root}/{save_loc}/results_eps_{eps}.csv"
        sorted_clusters_path = f"{root}/{save_loc}/sorted"
        semdedup_pruning_tables_path = f"{root}/{save_loc}/dataframes"
        os.makedirs(semdedup_pruning_tables_path, exist_ok=True)
        kept, removed, total = extract_pruned_data(
            id_col=id_col,
            id_type=id_type,
            sorted_clusters_path=sorted_clusters_path,
            semdedup_pruning_tables_path=semdedup_pruning_tables_path,
            eps=eps,
            num_clusters=semdedup_config.clustering["num_clusters"],
            output_csv_path=output_csv_path,
        )
        kept_list.append(kept)
        removed_list.append(removed)
        total_list.append(total)

    result_dict = {
        "eps": eps_list,
        "kept": kept_list,
        "removed": removed_list,
        "total": total_list,
    }
    df = pd.DataFrame(result_dict)
    summary_file = f"{root}/{save_loc}/summary.csv"
    df.to_csv(summary_file, index=False)


if __name__ == "__main__":
    config_file = "configs/config.yaml"
    semdedup_config = SemDedupConfig.from_yaml(config_file)
    root = semdedup_config.cache_dir
    save_loc = semdedup_config.clustering["save_loc"]

    logger = create_logger(
        rank=0,
        log_file=f"{root}/{save_loc}/extract_dedup_data.log",
        name="logger-extract-dedup-data",
        log_level=logging.INFO,
        stdout=True,
    )

    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")
    extract_dedup_data(semdedup_config=semdedup_config, logger=logger)
    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")
