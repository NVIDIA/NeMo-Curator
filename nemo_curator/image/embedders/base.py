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
from abc import ABC, abstractmethod

import dask.dataframe as dd

from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.utils.distributed_utils import load_object_on_worker


class ImageEmbedder(ABC):
    def __init__(self, image_embedding_column) -> None:
        self.image_embedding_column = image_embedding_column

    def __call__(self, dataset: ImageTextPairDataset) -> ImageTextPairDataset:
        meta_df = dataset.metadata._meta.copy()
        meta_df[self.image_embedding_column] = [1.0, 2.0]
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
            "image_embedding_model", self.load_embedding_model
        )

        total_samples = pipeline.epoch_size()
        samples_completed = 0
        while samples_completed < total_samples:
            image, text, meta = pipeline.run()

            print(f"Image: {image}. Text: {text}. Meta: {meta}")
            break
            embeddings = model(image)

        partition[self.image_embedding_column] = [[1.234]] * len(partition)
        return partition

    @abstractmethod
    def load_data_pipline(self, tar_path: str):
        pass

    @abstractmethod
    def load_embedding_model(self):
        pass
