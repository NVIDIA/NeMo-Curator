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
import os
from abc import ABC, abstractmethod
from typing import Iterable

import cupy as cp
import torch

from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.utils.cudf_utils import create_list_series_from_1d_or_2d_ar
from nemo_curator.utils.distributed_utils import load_object_on_worker


class ImageEmbedder(ABC):
    def __init__(self, model_name: str, image_embedding_column: str) -> None:
        self.model_name = model_name
        self.image_embedding_column = image_embedding_column

    def __call__(self, dataset: ImageTextPairDataset) -> ImageTextPairDataset:
        meta = dataset.metadata.dtypes.to_dict()
        meta[self.image_embedding_column] = "object"
        embedding_df = dataset.metadata.map_partitions(
            self._run_inference, dataset.tar_files, meta=meta
        )

        return ImageTextPairDataset(
            dataset.path, metadata=embedding_df, tar_files=dataset.tar_files
        )

    def _run_inference(self, partition, tar_paths, partition_info=None):
        tar_path = tar_paths[partition_info["number"]]
        device_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

        model = load_object_on_worker(
            self.model_name,
            self.load_embedding_model,
            {"device": f"cuda:{device_id}"},
        )

        dataset = self.load_dataset_shard(tar_path, device_id=device_id)
        final_image_embeddings = []
        samples_completed = 0
        with torch.no_grad():
            for batch in dataset:
                image_embeddings = model(batch)
                final_image_embeddings.append(image_embeddings)

                batch_size = len(image_embeddings)
                samples_completed += batch_size

                print(
                    f"{tar_path} - Embedding Creation with {self.model_name} Samples Completed: {samples_completed}."
                )

        if samples_completed != len(partition):
            raise RuntimeError(
                f"Mismatch in sample count for partition {partition_info['number']}. "
                f"{len(partition)} samples found in the metadata, but {samples_completed} found in {tar_path}."
            )

        concat_output = cp.asarray(torch.cat(final_image_embeddings, dim=0))
        partition[self.image_embedding_column] = create_list_series_from_1d_or_2d_ar(
            concat_output, index=partition.index
        )

        return partition

    @abstractmethod
    def load_dataset_shard(self, tar_path: str) -> Iterable:
        pass

    @abstractmethod
    def load_embedding_model(self, device):
        pass
