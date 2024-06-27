# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def extract_pruned_data(
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

        cluster_i = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
        )

        df_cluster_i = pd.DataFrame(
            {
                "adlr_id": cluster_i[:, 0],
                "global_id": cluster_i[:, 1],
                "dist": cluster_i[:, 2],
                "cluster": cluster_i[:, 3],
            }
        )
        df_cluster_i.adlr_id = df_cluster_i.adlr_id.astype("str")
        df_cluster_i.global_id = df_cluster_i.global_id.astype("int32")
        df_cluster_i.dist = df_cluster_i.dist.astype("float32")
        df_cluster_i.cluster = df_cluster_i.cluster.astype("int32")
        total += df_cluster_i.shape[0]

        print("==>", semdedup_pruning_tables_path, cluster_id)
        with open(
            f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
        ) as file:
            semdedup_pruning_tables = pickle.load(file)

        if semdedup_pruning_tables.shape[0] == 1:
            print("[]", cluster_id, semdedup_pruning_tables.shape, df_cluster_i.shape)
            continue

        ## -- See which examples to keep in this cluster.
        ## -- semdedup_pruning_tables contain True values for the examples to be removed.

        items_to_keep = semdedup_pruning_tables[
            semdedup_pruning_tables[f"eps={eps}"] == False
        ]["global_id"].tolist()

        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]

        ## -- retrieve only the examples we want and add to the list.
        dedup_cluster = df_cluster_i[df_cluster_i.global_id.isin(items_to_keep)]
        dedup_clusters.append(dedup_cluster)

    result = pd.concat(dedup_clusters)
    result.to_csv(output_csv_path, index=False)
    num_removed = total - result.shape[0]

    print(f"DONE saving {result.shape[0]} out of {total}. Removed: {num_removed}")
    return


if __name__ == "__main__":
    config_file = "./configs_cf.yml"
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    root = params["root"]
    save_loc = params["clustering"]["save_loc"]
    eps = params["extract_dedup"]["eps"]

    output_csv_path = f"{root}/{save_loc}/results_eps_{eps}.csv"
    sorted_clusters_path = f"{root}/{save_loc}/sorted"
    semdedup_pruning_tables_path = f"{root}/{save_loc}/dataframes"
    os.makedirs(semdedup_pruning_tables_path, exist_ok=True)
    extract_pruned_data(
        sorted_clusters_path=sorted_clusters_path,
        semdedup_pruning_tables_path=semdedup_pruning_tables_path,
        eps=eps,
        num_clusters=params["clustering"]["num_clusters"],
        output_csv_path=output_csv_path,
    )
