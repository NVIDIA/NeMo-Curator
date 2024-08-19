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
from typing import Optional

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import open_clip
from nvidia.dali import pipeline_def

from nemo_curator.image.embedders.base import ImageEmbedder


class OpenClipImageEmbedder(ImageEmbedder):
    def __init__(
        self,
        model_name: str,
        pretrained: Optional[str] = None,
        batch_size: int = 1,
        num_threads_per_worker=4,
        image_embedding_column="image_embedding",
    ) -> None:
        super().__init__(image_embedding_column=image_embedding_column)

        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.num_threads_per_worker = num_threads_per_worker

    def load_data_pipline(self, tar_path: str):
        # Create the DALI pipeline
        @pipeline_def(
            batch_size=self.batch_size,
            num_threads=self.num_threads_per_worker,
        )
        def webdataset_pipeline(_tar_path: str):
            img_raw, text, json = fn.readers.webdataset(
                paths=_tar_path,
                ext=["jpg", "txt", "json"],
                missing_component_behavior="error",
            )
            img = fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)
            img = fn.crop_mirror_normalize(
                img,
                dtype=types.FLOAT,
                mean=[0, 0, 0],
                std=[255, 255, 255],
            )

            resized = fn.resize(img, device="gpu", resize_shorter=224)
            output = fn.crop_mirror_normalize(
                resized,
                dtype=types.FLOAT,
                crop=(224, 224),
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            return output, text, json

        return webdataset_pipeline(tar_path)

    def load_embedding_model(self, device="cuda"):
        return open_clip.create_model(
            self.model_name, pretrained=self.pretrained, device=device
        )
