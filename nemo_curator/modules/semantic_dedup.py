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
import shutil
from dataclasses import dataclass
from typing import Union

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from cuml.dask.cluster import KMeans
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import write_to_disk


# Embedding Creation Module
@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    max_mem_gb: int
    max_seq_length: int = None

    def __post_init__(self):
        self.max_seq_length = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        ).model_max_length
        # Gaurd against the HF bug
        # which sets max_seq_length to max(int) for some models
        if self.max_seq_length > 1e5:
            self.max_seq_length = AutoConfig.from_pretrained(
                self.model_name_or_path
            ).max_position_embeddings


class EmbeddingPytorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path, config=self.config
        )

    def feature(self, input_ids, attention_mask):
        with torch.autocast(device_type=input_ids.device.type):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return embeddings

    @torch.no_grad()
    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        return self._mean_pooling(feature, batch["attention_mask"])

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)


class EmbeddingCrossFitModel(HFModel):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        super().__init__(
            self.config.model_name_or_path, max_mem_gb=self.config.max_mem_gb
        )

    def load_model(self, device="cuda"):
        model = EmbeddingPytorchModel(self.config)
        model = model.to(device)
        model.eval()
        return model

    def max_seq_length(self):
        return self.config.max_seq_length

    def load_config(self):
        return AutoConfig.from_pretrained(self.model_name_or_path)


class EmbeddingCreator:
    def __init__(
        self,
        model_name_or_path,
        max_memory,
        batch_size,
        embedding_output_dir,
        logger: Union[logging.Logger, str] = "./",
    ):
        self.embeddings_config = EmbeddingConfig(
            model_name_or_path=model_name_or_path, max_mem_gb=max_memory
        )
        self.batch_size = batch_size
        self.logger = self._setup_logger(logger)
        self.embedding_output_dir = embedding_output_dir
        self.model = EmbeddingCrossFitModel(self.embeddings_config)

    def _setup_logger(self, logger):
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="compute-embeddings",
                log_file=os.path.join(logger, "compute_embeddings.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def create_embeddings(
        self, ddf: dask_cudf.DataFrame, input_column="text"
    ) -> dask_cudf.DataFrame:
        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[input_column],
                tokenizer_type="sentencepiece",
                max_length=self.embeddings_config.max_seq_length,
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col="embeddings",
            ),
            keep_cols=ddf.columns.tolist(),
        )
        return pipe(ddf)

    def __call__(
        self, dataset: DocumentDataset, input_column="text"
    ) -> DocumentDataset:
        embedding_ddf = self.create_embeddings(dataset.df, input_column)
        write_to_disk(
            embedding_ddf,
            self.embedding_output_dir,
            write_to_filename=True,
            output_type="parquet",
        )
        return DocumentDataset(
            dask_cudf.read_parquet(
                self.embedding_output_dir, blocksize="2GB", aggregate_files=True
            )
        )


### Clustering Module
def get_embedding_ar(df: "cudf.DataFrame") -> cp.ndarray:
    return df["embeddings"].list.leaves.values.reshape(len(df), -1)


def add_dist_to_cents(df: "cudf.DataFrame", centroids: cp.ndarray) -> "cudf.DataFrame":
    embed_array = get_embedding_ar(df)
    centroids_ar = centroids[df["nearest_cent"].values]
    dist_to_cents = cp.sqrt(np.sum((embed_array - centroids_ar) ** 2, axis=1))
    df["dist_to_cent"] = dist_to_cents
    return df


class ClusteringModel:
    def __init__(
        self,
        max_iter,
        n_clusters,
        clustering_output_dir,
        logger: Union[logging.Logger, str] = "./",
    ):
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.clustering_output_dir = clustering_output_dir
        self.logger = self._setup_logger(logger)

        if not os.path.exists(self.clustering_output_dir):
            os.makedirs(self.clustering_output_dir)
        else:
            self.logger.warning(
                f"Clustering output directory {self.clustering_output_dir} already exists and will be overwritten"
            )

    def _setup_logger(self, logger):
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="compute-clusters",
                log_file=os.path.join(logger, "compute_clusters.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def __call__(self, embeddings_df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:

        assert "embeddings" in embeddings_df.columns

        embeddings_df = embeddings_df.to_backend("pandas").persist()
        embeddings_df = embeddings_df.repartition(partition_size="2GB")
        embeddings_df = embeddings_df.to_backend("cudf")

        cupy_darr = embeddings_df.map_partitions(
            get_embedding_ar, meta=cp.ndarray([1, 1])
        )
        cupy_darr.compute_chunk_sizes()

        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        self.logger.info("KMeans starting fit")
        kmeans.fit(cupy_darr)
        self.logger.info("KMeans fit complete")

        self.logger.info(
            "Computing nearest centroids + distance to centers using kmeans.predict"
        )
        nearest_cents = kmeans.predict(cupy_darr)
        embeddings_df["nearest_cent"] = nearest_cents.astype(np.int32)

        meta_df = embeddings_df._meta.copy()
        meta_df["dist_to_cent"] = cp.zeros(1)
        embeddings_df = embeddings_df.map_partitions(
            add_dist_to_cents, centroids=kmeans.cluster_centers_, meta=meta_df
        )
        centroids = kmeans.cluster_centers_
        embeddings_df = embeddings_df.reset_index(drop=True)
        kmeans_centroids_file = os.path.join(
            self.clustering_output_dir, "kmeans_centroids.npy"
        )
        np.save(kmeans_centroids_file, centroids)
        self.logger.info("Saving centroids complete")

        output_dir = os.path.join(self.clustering_output_dir, "embs_by_nearest_center")
        if os.path.exists(output_dir):
            self.logger.warning(
                f"Output directory {output_dir} already exists and will be overwritten"
            )
            shutil.rmtree(output_dir)
        embeddings_df.to_parquet(
            output_dir,
            index=False,
            partition_on="nearest_cent",
        )
        self.logger.info(f"Saved embeddings by nearest center to {output_dir}")

        fps = [
            os.path.join(output_dir, file_name) for file_name in os.listdir(output_dir)
        ]
        embeddings_df = dd.from_map(cudf.read_parquet, fps)
        return embeddings_df
