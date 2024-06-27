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
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from utils import get_logger


def extract_pruned_data(
    id_col,
    id_type,
    sorted_clusters_path,
    semdedup_pruning_tables_path,
    eps,
    num_clusters,
    output_csv_path,
):

    ## -- duplicates we want to keep/remove.
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

        logger.info(
            f"semdedup_pruning_tables_path: {semdedup_pruning_tables_path}, cluster_id: {cluster_id}"
        )

        with open(
            f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
        ) as file:
            semdedup_pruning_tables = pickle.load(file)

        if semdedup_pruning_tables.shape[0] == 1:
            logger.info(
                f"""cluster_id: {cluster_id},
                    semdedup_pruning_tables.shape: {semdedup_pruning_tables.shape},
                    df_cluster_i.shape: {df_cluster_i.shape}"""
            )
            continue

        ## -- See which examples to keep in this cluster.
        ## -- semdedup_pruning_tables contain True values for the examples to be removed.

        items_to_keep = semdedup_pruning_tables[
            semdedup_pruning_tables[f"eps={eps}"] == False
        ]["id"].tolist()

        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]

        ## -- retrieve only the examples we want and add to the list.
        dedup_cluster = df_cluster_i[df_cluster_i[id_col].isin(items_to_keep)]
        dedup_clusters.append(dedup_cluster)

    result = pd.concat(dedup_clusters)
    result.to_csv(output_csv_path, index=False)
    num_removed = total - result.shape[0]

    logger.info(f"DONE saving {result.shape[0]} out of {total}. Removed: {num_removed}")
    return result.shape[0], num_removed, total


if __name__ == "__main__":
    config_file = "./configs_cf.yml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    root = params["root"]
    save_loc = params["clustering"]["save_loc"]

    logger = get_logger(
        file_name=f"{root}/{save_loc}/extract_dedup_data.log",
        level=logging.INFO,
        stdout=True,
    )

    if params["extract_dedup"]["use_eps_from_yml"]:
        eps = params["extract_dedup"]["eps"]
        eps_list = [float(x) for x in eps.split(" ")]
    else:
        eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
        eps_list2 = [0.1 + x * 0.005 for x in range(34)]
        eps_list = eps_list1 + eps_list2
    kept_list = []
    removed_list = []
    total_list = []
    id_col = params["id_col"]["name"]
    id_type = params["id_col"]["type"]

    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")

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
            num_clusters=params["clustering"]["num_clusters"],
            output_csv_path=output_csv_path,
        )
        kept_list.append(kept)
        removed_list.append(removed)
        total_list.append(total)

    dict = {
        "eps": eps_list,
        "kept": kept_list,
        "removed": removed_list,
        "total": total_list,
    }
    df = pd.DataFrame(dict)
    summary_file = f"{root}/{save_loc}/summary.csv"
    df.to_csv(summary_file, index=False)

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")
