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
        text_field: str = "text",
        id_field: str = "id",
        id_field_type: str = "int",
        logger: Union[logging.Logger, str] = "./",
    ) -> None:
        """
        Initialize the SemDedup class.

        Args:
            config (SemDedupConfig): Configuration for SemDedup.
            input_column (str): Column name from the data to be used for embedding generation.
                Default is "text".
            id_column (str): Column name used as the identifier in the dataset.
                Default is "id".
            id_column_type (str): Data type of id_column. Default is "int".
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
        """
        super().__init__(input_backend="cudf")
        self.config = config
        self.logger = logger
        cache_dir = config.cache_dir
        self.embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path=config.embedding_model_name_or_path,
            embedding_batch_size=config.embedding_batch_size,
            embedding_output_dir=os.path.join(cache_dir, config.embeddings_save_loc),
            embedding_max_mem_gb=config.embedding_max_mem_gb,
            embedding_pooling_strategy=config.embedding_pooling_strategy,
            text_field=text_field,
            embedding_column=config.embedding_column,
            write_embeddings_to_disk=config.write_embeddings_to_disk,
            write_to_filename=config.write_to_filename,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.clustering_model = ClusteringModel(
            id_field=id_field,
            max_iter=config.max_iter,
            n_clusters=config.n_clusters,
            clustering_output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            embedding_column=config.embedding_column,
            sim_metric=config.sim_metric,
            which_to_keep=config.which_to_keep,
            sort_clusters=config.sort_clusters,
            kmeans_with_cos_dist=config.kmeans_with_cos_dist,
            clustering_input_partition_size=config.clustering_input_partition_size,
            logger=logger,
            profile_dir=self.config.profile_dir,
        )
        self.semantic_cluster_dedup = SemanticClusterLevelDedup(
            n_clusters=config.n_clusters,
            emb_by_clust_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "embs_by_nearest_center"
            ),
            sorted_clusters_dir=os.path.join(
                cache_dir, config.clustering_save_loc, "sorted"
            ),
            id_field=id_field,
            id_field_type=id_field_type,
            which_to_keep=config.which_to_keep,
            output_dir=os.path.join(cache_dir, config.clustering_save_loc),
            embedding_column=config.embedding_column,
            logger=logger,
            profile_dir=self.config.profile_dir,
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
