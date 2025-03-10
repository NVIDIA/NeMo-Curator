# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import time
from typing import List, Optional, Union

import dask.bag as db

from nemo_curator.cache import Cache
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.semdedup_utils import (
    extract_dedup_data,
    get_semantic_matches_per_cluster,
)


class SemanticClusterLevelDedup:
    def __init__(
        self,
        n_clusters: int = 1000,
        id_column: str = "id",
        id_column_type: str = "int",
        which_to_keep: str = "hard",
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        embedding_column: str = "embeddings",
        clustering_save_loc: str = "clustering_results",
        logger: Union[logging.Logger, str] = "./",
        profile_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the SemanticClusterLevelDedup class.

        Args:
            n_clusters (int): Number of clusters. Default is 1000.
            id_column (str): Column name used as the identifier in the dataset.
                Default is "id".
            id_column_type (str): Data type of id_column. Default is "int".
            which_to_keep (str): Method to determine which duplicates to keep.
                Default is "hard".
            output_dir (str, optional): Directory to save output files.
                If None, it will be saved to cache_dir/clustering_save_loc.
                Default is None.
            cache_dir (str, optional): Should be the same as specified in ClusteringModel.
            embedding_column (str): The column name that stores the embeddings.
                Default is "embeddings".
            clustering_save_loc (str): Should be the same as specified in ClusteringModel.
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
            profile_dir (Optional[str]): If specified, directory to write Dask profile.
                Default is None.

        """
        self.n_clusters = n_clusters
        self.id_col = id_column
        self.id_col_type = id_column_type
        self.which_to_keep = which_to_keep
        self.computed_semantic_match_dfs = False
        self.embedding_column = embedding_column
        self.logger = self._setup_logger(logger)
        self.profile_dir = profile_dir

        if cache_dir is None:
            if Cache().get_cache_directory() is None:
                raise RuntimeError(
                    "No cache directory specified. Please initialize with Cache(cache_dir=...) "
                    "or SemanticClusterLevelDedup(cache_dir=...)"
                )
            else:
                cache_dir = Cache().get_cache_directory()
        self.emb_by_clust_dir = os.path.join(
            cache_dir, clustering_save_loc, "embs_by_nearest_center"
        )
        self.sorted_clusters_dir = os.path.join(
            cache_dir, clustering_save_loc, "sorted"
        )

        if output_dir is None:
            self.output_dir = os.path.join(cache_dir, clustering_save_loc)
        else:
            self.output_dir = output_dir
        self.semdedup_pruning_tables_dir = os.path.join(
            output_dir, "semdedup_pruning_tables"
        )

    def _setup_logger(self, logger: Union[logging.Logger, str]) -> logging.Logger:
        """
        Set up the logger.

        Args:
            logger (Union[logging.Logger, str]): Logger instance or path to the log file directory.

        Returns:
            logging.Logger: Configured logger.
        """
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="SemanticClusterLevelDedup",
                log_file=os.path.join(logger, "SemanticClusterLevelDedup.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def compute_semantic_match_dfs(
        self, eps_list: Optional[List[float]] = None
    ) -> None:
        """
        Compute semantic match dataframes for clusters.

        Args:
            eps_list (Optional[List[float]]): List of epsilon values for clustering.
        """
        if eps_list is None:
            eps_list1 = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
            eps_list2 = [0.1 + x * 0.005 for x in range(34)]
            eps_list = eps_list1 + eps_list2

        if os.path.exists(self.semdedup_pruning_tables_dir):
            self.logger.info(
                f"Removing existing directory {self.semdedup_pruning_tables_dir}"
            )
            shutil.rmtree(self.semdedup_pruning_tables_dir)
        expand_outdir_and_mkdir(self.semdedup_pruning_tables_dir)

        t0 = time.time()

        with performance_report_if_with_ts_suffix(
            self.profile_dir, "semantic-match-compute"
        ):
            tasks = db.from_sequence(
                list(range(self.n_clusters)), npartitions=self.n_clusters
            ).map(
                lambda cluster_id: get_semantic_matches_per_cluster(
                    cluster_id=cluster_id,
                    emb_by_clust_dir=self.emb_by_clust_dir,
                    sorted_clusters_dir=self.sorted_clusters_dir,
                    id_col=self.id_col,
                    id_col_type=self.id_col_type,
                    eps_list=eps_list,
                    output_dir=self.semdedup_pruning_tables_dir,
                    embedding_col=self.embedding_column,
                    which_to_keep=self.which_to_keep,
                )
            )
            tasks.compute()
        self.logger.info(
            f"Time taken for Computing Semantic Matches : {time.time() - t0}"
        )
        self.computed_semantic_match_dfs = True

    def extract_dedup_data(self, eps_to_extract: float) -> DocumentDataset:
        """
        Extract deduplicated data based on epsilon value.

        Args:
            eps_to_extract (float): Epsilon threshold for extracting deduplicated data.

        Returns:
            DocumentDataset: Dataset containing deduplicated documents.
        """
        if not self.computed_semantic_match_dfs:
            raise ValueError(
                "Run compute_semantic_match_dfs before calling extract_dedup_data"
            )

        output_summary_file = os.path.join(
            self.output_dir, f"dedup_summary_{eps_to_extract}.csv"
        )
        output_parquet_path = os.path.join(
            self.output_dir, f"unique_ids_{eps_to_extract}.parquet"
        )
        extract_dedup_data(
            eps=eps_to_extract,
            n_clusters=self.n_clusters,
            id_col=self.id_col,
            id_col_type=self.id_col_type,
            sorted_clusters_dir=self.sorted_clusters_dir,
            semdedup_pruning_tables_dir=self.semdedup_pruning_tables_dir,
            output_summary_file=output_summary_file,
            output_parquet_path=output_parquet_path,
            logger=self.logger,
            profile_dir=self.profile_dir,
        )

        fps = [
            os.path.join(output_parquet_path, file_name)
            for file_name in os.listdir(output_parquet_path)
        ]
        return DocumentDataset.read_parquet(fps, backend="cudf")
