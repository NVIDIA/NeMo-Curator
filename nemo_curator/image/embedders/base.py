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
import json
from abc import ABC, abstractmethod

import dask.dataframe as dd
import numpy as np
import torch
from nvidia.dali.plugin.pytorch import feed_ndarray

from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.utils.distributed_utils import load_object_on_worker


class ImageEmbedder(ABC):
    def __init__(self, image_embedding_column) -> None:
        self.image_embedding_column = image_embedding_column

    def __call__(self, dataset: ImageTextPairDataset) -> ImageTextPairDataset:
        meta_df = dataset.metadata._meta.copy()
        meta_df[self.image_embedding_column] = [[1.0]]
        embedding_df = dataset.metadata.map_partitions(
            self.inference, dataset.tar_files, meta=meta_df
        )

        return ImageTextPairDataset(
            dataset.path, metadata=embedding_df, tar_files=dataset.tar_files
        )

    def inference(self, partition, tar_paths, partition_info=None):
        tar_path = tar_paths[partition_info["number"]]
        pipeline = self.load_data_pipline(tar_path)
        pipeline.build()
        model = load_object_on_worker(
            "image_embedding_model", self.load_embedding_model, {}
        )

        total_samples = pipeline.epoch_size()
        total_samples = total_samples[list(total_samples.keys())[0]]
        samples_completed = 0
        final_image_embeddings = []
        while samples_completed < total_samples:
            image, text, meta = pipeline.run()

            print(f"Image: {image}. Text: {text}. Meta: {meta}")
            image = image.as_tensor()

            image_torch = torch.empty(
                image.shape(), dtype=torch.float32, device=self.device
            )
            feed_ndarray(image, image_torch)  # COPY !!!

            image = image_torch
            captions = [text.at(i).tostring().decode("utf-8") for i in range(len(text))]
            metadata = [
                json.loads(meta.at(i).tostring().decode("utf-8"))
                for i in range(len(meta))
            ]

            with torch.no_grad():
                image_features = model(image)
                batch_image_embeddings = np.asarray(
                    self.normalized(image_features.detach().cpu())
                )

            for embedding in batch_image_embeddings:
                final_image_embeddings.append(embedding)

        partition[self.image_embedding_column] = final_image_embeddings
        return partition

    @staticmethod
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    @abstractmethod
    def load_data_pipline(self, tar_path: str):
        pass

    @abstractmethod
    def load_embedding_model(self):
        pass
