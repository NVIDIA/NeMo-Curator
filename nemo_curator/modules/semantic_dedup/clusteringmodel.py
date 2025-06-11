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
import time
from typing import Dict, Optional, Union

import cudf
import cupy as cp
import dask.dataframe as dd
import numpy as np
from cuml.dask.cluster import KMeans

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import performance_report_if_with_ts_suffix
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_fs
from nemo_curator.utils.semdedup_utils import (
    COSINE_DIST_TO_CENT_COL,
    L2_DIST_TO_CENT_COL,
    add_l2_cosine_dist_to_centroid,
    get_array_from_df,
    normalize_embeddings_col_in_df,
)


# Clustering module
class ClusteringModel:
    def __init__(
        self,
        id_column: str = "id",
        max_iter: int = 100,
        n_clusters: int = 1000,
        clustering_output_dir: str = "./clustering_results",
        embedding_column: str = "embeddings",
        random_state: int = 1234,
        clustering_input_partition_size: Optional[str] = "2gb",
        logger: Union[logging.Logger, str] = "./",
        profile_dir: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the ClusteringModel with the provided settings for semantic clustering to help semantic deduplication.

        Args:
            id_column (str): Column name used as the identifier in the dataset.
                Default is "id".
            max_iter (int): Maximum iterations for clustering. The more iterations, the better the clustering.
                Default is 100.
            n_clusters (int): Number of clusters. Default is 1000.
            clustering_output_dir (str): Location to save clustering results.
                Default is "./clustering_results".
            embedding_column (str): The column name that stores the embeddings.
                Default is "embeddings".
            random_state (int): KMeans random state used for reproducibility.
                Default is 1234.
            clustering_input_partition_size (Optional[str]): The size of data partition with which to run KMeans.
                Default is "2gb". If None, then the dataset is not repartitioned.
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
            profile_dir (Optional[str]): If specified, directory to write Dask profile.
                Default is None.
            storage_options (Optional[Dict[str, str]]): Storage options for the file system.
                Default is None.
        """
        self.id_col = id_column
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.embedding_column = embedding_column
        self.random_state = random_state
        self.clustering_input_partition_size = clustering_input_partition_size
        self.logger = self._setup_logger(logger)
        self.profile_dir = profile_dir
        self.storage_options = storage_options
        self.fs = get_fs(self.clustering_output_dir, self.storage_options)

        print(f"Clustering output directory: {self.clustering_output_dir}")
        print(f"Storage options: {self.storage_options}")
        print(f"File system: {self.fs}")

        if not self.fs.exists(self.clustering_output_dir):
            expand_outdir_and_mkdir(
                self.clustering_output_dir, self.storage_options, self.fs
            )
        else:
            self.logger.warning(
                f"Clustering output directory {self.clustering_output_dir} already exists and will be overwritten"
            )

    def _setup_logger(self, logger):
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

    def __call__(self, embeddings_dataset: DocumentDataset):
        embeddings_df = embeddings_dataset.df

        if self.embedding_column not in embeddings_df.columns:
            raise ValueError(
                f'Expected embedding column "{self.embedding_column}"'
                f" to be in dataset. Only found columns {embeddings_df.columns}"
            )

        with performance_report_if_with_ts_suffix(self.profile_dir, "clustering-model"):
            embeddings_df = embeddings_df[[self.id_col, self.embedding_column]]
            if self.clustering_input_partition_size is not None:
                embeddings_df = embeddings_df.repartition(
                    partition_size=self.clustering_input_partition_size
                )

            try:
                embeddings_df = embeddings_df.to_backend("pandas").persist()
                embeddings_length = embeddings_df.shape[0].compute()

                if embeddings_length < self.n_clusters:
                    raise ValueError(
                        "Number of clusters is greater than the number of documents in your dataset: "
                        f"dataset length is {embeddings_length} while n_clusters is set to {self.n_clusters}. "
                        f"Please reduce n_clusters to be less than or equal to {embeddings_length}."
                    )
            except IndexError as e:
                raise IndexError(
                    f'Original error message: "{e}". '
                    "This could be due to empty partitions in your DocumentDataset. "
                    "Please check your dataset for empty partitions and remove them if necessary."
                )

            embeddings_df = embeddings_df.to_backend("cudf")
            # Normalize embeddings before clustering
            embeddings_df = embeddings_df.map_partitions(
                normalize_embeddings_col_in_df,
                embedding_col=self.embedding_column,
                meta=embeddings_df._meta.copy(),
            )
            cupy_normalized_darr = embeddings_df.map_partitions(
                get_array_from_df, self.embedding_column, meta=cp.ndarray([1, 1])
            )
            cupy_normalized_darr.compute_chunk_sizes()

            # Perform KMeans clustering (KMeans.fit)
            t0 = time.time()
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=1,
            )
            self.logger.info("KMeans starting fit")
            kmeans.fit(cupy_normalized_darr)
            self.logger.info("KMeans fit complete")
            self.logger.info(f"Time taken for KMeans fit: {time.time() - t0}")
            # Compute nearest centroids using kmeans.predict
            self.logger.info(
                "Computing nearest centroids and distance to centers using kmeans.predict"
            )
            t0 = time.time()
            nearest_cents = kmeans.predict(cupy_normalized_darr)
            self.logger.info(f"Time taken for KMeans predict: {time.time() - t0}")
            t0 = time.time()
            embeddings_df["nearest_cent"] = nearest_cents.astype(np.int32)
            del nearest_cents
            # Add L2 and cosine distance to centroid columns to the dataframe
            meta_df_with_l2_dist = embeddings_df._meta.copy()
            meta_df_with_l2_dist[L2_DIST_TO_CENT_COL] = cp.zeros(1)
            meta_df_with_l2_dist[COSINE_DIST_TO_CENT_COL] = cp.zeros(1)
            embeddings_df = embeddings_df.map_partitions(
                add_l2_cosine_dist_to_centroid,
                embedding_col=self.embedding_column,
                centroids=kmeans.cluster_centers_,
                meta=meta_df_with_l2_dist,
            )
            embeddings_df = embeddings_df.reset_index(drop=True)
            # Save centroids to a file
            centroids = kmeans.cluster_centers_
            kmeans_centroids_file = os.path.join(
                self.clustering_output_dir, "kmeans_centroids.npy"
            )
            with self.fs.open(kmeans_centroids_file, "wb") as f:
                np.save(f, centroids)
            self.logger.info("Saving centroids complete")
            del kmeans, cupy_normalized_darr, centroids

            # Save embeddings by nearest center to a file
            embeddings_by_nearest_center_dir = os.path.join(
                self.clustering_output_dir, "embs_by_nearest_center"
            )
            if self.fs.exists(embeddings_by_nearest_center_dir):
                self.logger.warning(
                    f"Output directory {embeddings_by_nearest_center_dir} already exists and will be overwritten"
                )
                self.fs.rm(embeddings_by_nearest_center_dir, recursive=True)

            embeddings_df.map_partitions(
                lambda df: df.to_parquet(
                    embeddings_by_nearest_center_dir,
                    index=False,
                    partition_cols=["nearest_cent"],
                    storage_options=self.storage_options,
                ),
                # we specify meta to allow skipping _meta_nonempty
                meta=dict(),
            ).compute()

            self.logger.info(
                f"Time taken for assigning distance to each embedding: {time.time() - t0}s"
                f" and output written at {embeddings_by_nearest_center_dir}"
            )

            del embeddings_df
        # We read this way to ensure each cluster is read in a single partition
        # This allows us to perform pairwise similarity within the cluster
        fps = [
            os.path.join(embeddings_by_nearest_center_dir, f"nearest_cent={i}")
            for i in range(self.n_clusters)
        ]
        embeddings_df = dd.from_map(
            cudf.read_parquet, fps, storage_options=self.storage_options
        )
        return DocumentDataset(embeddings_df)
