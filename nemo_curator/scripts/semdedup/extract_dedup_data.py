import logging
import os
from datetime import datetime

from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup import SemanticClusterLevelDedup
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main():
    semdedup_config = SemDedupConfig.from_yaml("config.yaml")
    parser = ArgumentHelper.parse_semdedup_args(add_input_args=False)
    args = parser.parse_args()

    root = semdedup_config.cache_dir
    save_loc = semdedup_config.clustering["save_loc"]
    client = get_client(**ArgumentHelper.parse_client_args(args))

    logger = create_logger(
        rank=0,
        log_file=os.path.join(root, save_loc, "extract_dedup_data.log"),
        name="logger-extract-dedup-data",
        log_level=logging.INFO,
        stdout=True,
    )

    dt1 = datetime.now()
    logger.info(f"Start: {dt1}")
    cache_dir = semdedup_config.cache_dir
    semantic_dedup = SemanticClusterLevelDedup(
        n_clusters=semdedup_config.clustering["n_clusters"],
        emb_by_clust_dir=os.path.join(
            cache_dir, semdedup_config.clustering["save_loc"], "embs_by_nearest_center"
        ),
        sorted_clusters_dir=os.path.join(
            cache_dir, semdedup_config.clustering["save_loc"], "sorted"
        ),
        id_col=semdedup_config.id_col["name"],
        id_col_type=semdedup_config.id_col["type"],
        which_to_keep=semdedup_config.semdedup["which_to_keep"],
        output_dir=os.path.join(
            semdedup_config.cache_dir, semdedup_config.clustering["save_loc"]
        ),
        logger=logger,
    )

    semantic_dedup.compute_semantic_match_dfs()
    for eps in semdedup_config.extract_dedup["eps"].split(" "):
        eps = float(eps)
        dedup_id_dataset = semantic_dedup.extract_dedup_data(eps=eps)
        print(dedup_id_dataset.df.head(10))

    dt2 = datetime.now()
    logger.info(f"End: {dt2}")
    elapse = (dt2 - dt1).total_seconds() / 60
    logger.info(f"elapse: {elapse}")

    client.cancel(client.futures, force=True)
    client.close()
    return


if __name__ == "__main__":
    main()
