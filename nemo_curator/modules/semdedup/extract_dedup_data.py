import logging
import os
from datetime import datetime
from typing import Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import progress

from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import parse_client_args, parse_semdedup_args


def get_num_records(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "rb") as f:
        # Read the header of the npy file
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape[0]


def _get_empty_results_df(id_col, id_type):
    meta_df = pd.DataFrame(
        {
            id_col: np.empty(0, dtype="int64"),
            "dist": np.empty(0, dtype="float32"),
            "cluster": np.empty(0, dtype="int32"),
        }
    )
    meta_df[id_col] = meta_df[id_col].astype(id_type)
    return meta_df


def process_single_cluster(
    cluster_id: int,
    id_col: str,
    id_type: str,
    sorted_clusters_path: str,
    semdedup_pruning_tables_path: str,
    eps: float,
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Processes data for a single cluster, applying pruning based on specified epsilon.

    Args:
        cluster_id (int): The specific cluster ID to process.
        id_col (str): The name of the ID column.
        id_type (str): The data type of the ID column.
        sorted_clusters_path (str): Path to the sorted clusters directory.
        semdedup_pruning_tables_path (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.

    Returns:
        Tuple[pd.DataFrame, int, int, int]: A DataFrame of the pruned cluster data,
        number of kept records, number of removed records, and total records.
    """
    sorted_fname = os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
    if not os.path.exists(sorted_fname):
        return _get_empty_results_df(id_col, id_type)

    cluster_data = np.load(sorted_fname)
    df_cluster = pd.DataFrame(
        {
            id_col: cluster_data[:, 0],
            "dist": cluster_data[:, 1],
            "cluster": cluster_data[:, 2],
        }
    )

    df_cluster[id_col] = df_cluster[id_col].astype(id_type)
    df_cluster["dist"] = df_cluster["dist"].astype("float32")
    df_cluster["cluster"] = df_cluster["cluster"].astype("int32")

    cluster_df_fname = os.path.join(
        semdedup_pruning_tables_path, f"cluster_{cluster_id}.parquet"
    )
    pruning_table = pd.read_parquet(cluster_df_fname)

    if pruning_table.shape[0] == 1:
        return df_cluster

    items_to_keep = pruning_table[pruning_table[f"eps={eps}"] == False]["id"].tolist()
    pruned_cluster = df_cluster[df_cluster[id_col].isin(items_to_keep)]

    return pruned_cluster


def extract_pruned_data(
    id_col: str,
    id_type: str,
    sorted_clusters_path: str,
    semdedup_pruning_tables_path: str,
    eps: float,
    n_clusters: int,
    output_parquet_path: str,
) -> Tuple[int, int, int]:
    """
    Extracts pruned data from sorted clusters and saves it to a CSV file.

    Args:
        id_col (str): The name of the ID column.
        id_type (str): The data type of the ID column.
        sorted_clusters_path (str): Path to the sorted clusters directory.
        semdedup_pruning_tables_path (str): Path to the pruning tables directory.
        eps (float): Epsilon value for pruning.
        n_clusters (int): Number of clusters.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        Tuple[int, int, int]: Number of kept records, removed records, and total records.
    """

    results_df = dd.from_map(
        process_single_cluster,
        range(n_clusters),
        id_col=id_col,
        id_type=id_type,
        sorted_clusters_path=sorted_clusters_path,
        semdedup_pruning_tables_path=semdedup_pruning_tables_path,
        eps=eps,
    )
    results_df = results_df.persist()
    progress(results_df)

    results_df.to_parquet(output_parquet_path)

    total_kept = len(results_df)

    np_files = [
        os.path.join(sorted_clusters_path, f"cluster_{i}.npy")
        for i in range(n_clusters)
    ]
    total_records = sum(get_num_records(file_path) for file_path in np_files)

    # Aggregate results
    total_removed = total_records - total_kept
    return total_kept, total_removed, total_records


def extract_dedup_data(semdedup_config: SemDedupConfig, logger: logging.Logger) -> None:
    """
    Extracts deduplicated data based on provided parameters and logs the process.

    Args:
        semdedup_config: Configuration object for SemDedup.
        logger (logging.Logger): Logger for logging the process.
    """

    root = semdedup_config.cache_dir
    save_loc = semdedup_config.clustering["save_loc"]

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
        output_parquet_path = f"{root}/{save_loc}/results_eps_{eps}.parquet"
        sorted_clusters_path = f"{root}/{save_loc}/sorted"
        semdedup_pruning_tables_path = f"{root}/{save_loc}/dataframes"
        os.makedirs(semdedup_pruning_tables_path, exist_ok=True)
        kept, removed, total = extract_pruned_data(
            id_col=id_col,
            id_type=id_type,
            sorted_clusters_path=sorted_clusters_path,
            semdedup_pruning_tables_path=semdedup_pruning_tables_path,
            eps=eps,
            n_clusters=semdedup_config.clustering["n_clusters"],
            output_parquet_path=output_parquet_path,
        )
        logger.info(
            f"DONE saving {kept} out of {total}. Removed: {removed}. Epsilon: {eps:.4f}"
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


def main():
    semdedup_config = SemDedupConfig.from_yaml("configs/config.yaml")
    parser = parse_semdedup_args(add_input_args=False)
    args = parser.parse_args()
    client = get_client(**parse_client_args(args))

    root = semdedup_config.cache_dir
    save_loc = semdedup_config.clustering["save_loc"]
    logger = create_logger(
        rank=0,
        log_file=os.path.join(root, save_loc, "extract_dedup_data.log"),
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

    client.cancel(client.futures, force=True)
    client.close()
    return


if __name__ == "__main__":
    main()
