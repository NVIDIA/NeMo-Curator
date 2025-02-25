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
from typing import Union

from nemo_curator.cache import Cache
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import BaseModule
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup.clusteringmodel import ClusteringModel
from nemo_curator.modules.semantic_dedup.embeddings import EmbeddingCreator
from nemo_curator.modules.semantic_dedup.semanticclusterleveldedup import (
    SemanticClusterLevelDedup,
)


class SemDedup(BaseModule):
    def __init__(
        self,
        config: SemDedupConfig,
        input_column: str = "text",
        id_column: str = "id",
        id_column_type: str = "int",
        logger: Union[logging.Logger, str] = "./",
    ) -> None:
        """
        Initialize the SemDedup class.

        Args:
            config (SemDedupConfig): Configuration for SemDedup.
            logger (Union[logging.Logger, str]): Logger instance or path to the log file directory.
        """

        super().__init__(input_backend="cudf")
        self.config = config
        self.logger = logger
        if config.cache_dir is not None:
            cache_dir = config.cache_dir
        elif Cache().get_cache_directory() is not None:
            cache_dir = Cache().get_cache_directory()
        else:
            raise RuntimeError(
                "No cache directory specified. Please initialize with Cache(cache_dir=...) "
                "or specify a cache_dir in your YAML file."
            )
        profile_dir = self.config.profile_dir
        clustering_save_loc = config.clustering_save_loc

        self.embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path=config.embedding_model_name_or_path,
            embedding_batch_size=config.embedding_batch_size,
            cache_dir=cache_dir,
            embeddings_save_loc=config.embeddings_save_loc,
            embedding_max_mem_gb=config.embedding_max_mem_gb,
            embedding_pooling_strategy=config.embedding_pooling_strategy,
            input_column=input_column,
            embedding_column=config.embedding_column,
            write_embeddings_to_disk=config.write_embeddings_to_disk,
            write_to_filename=config.write_to_filename,
            logger=logger,
            profile_dir=profile_dir,
        )
        self.clustering_model = ClusteringModel(
            id_column=id_column,
            max_iter=config.max_iter,
            n_clusters=config.n_clusters,
            cache_dir=cache_dir,
            clustering_save_loc=clustering_save_loc,
            embedding_column=config.embedding_column,
            sim_metric=config.sim_metric,
            which_to_keep=config.which_to_keep,
            sort_clusters=config.sort_clusters,
            kmeans_with_cos_dist=config.kmeans_with_cos_dist,
            clustering_input_partition_size=config.clustering_input_partition_size,
            logger=logger,
            profile_dir=profile_dir,
        )
        self.semantic_cluster_dedup = SemanticClusterLevelDedup(
            n_clusters=config.n_clusters,
            id_column=id_column,
            id_column_type=id_column_type,
            which_to_keep=config.which_to_keep,
            cache_dir=cache_dir,
            embedding_column=config.embedding_column,
            clustering_save_loc=clustering_save_loc,
            logger=logger,
            profile_dir=profile_dir,
            # Hardcoded path
            output_dir=os.path.join(cache_dir, clustering_save_loc),
        )
        self.eps_thresholds = config.eps_thresholds
        self.eps_to_extract = config.eps_to_extract

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Execute the SemDedup process.

        Args:
            dataset (DocumentDataset): Input dataset for deduplication.

        Returns:
            DocumentDataset: Deduplicated dataset.
        """
        embeddings_dataset = self.embedding_creator(dataset)
        self.clustering_model(embeddings_dataset)
        self.semantic_cluster_dedup.compute_semantic_match_dfs(self.eps_thresholds)
        return self.semantic_cluster_dedup.extract_dedup_data(
            eps_to_extract=self.eps_to_extract
        )
