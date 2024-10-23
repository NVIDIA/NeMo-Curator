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
from typing import List

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import Module
from nemo_curator.utils.module_utils import is_batched


class DocumentModifier(Module, ABC):
    def __init__(
        self,
        text_fields: List[str] = ["text"],
        meta=(None, str),
        input_backend: str = "pandas",
    ):
        super().__init__(input_backend=input_backend)
        self.text_fields = text_fields
        self.meta = meta

    @abstractmethod
    def modify_document(self, text):
        raise NotImplementedError(
            "score_document method must be implemented by subclasses"
        )

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        text_fields = (
            self.text_fields if len(self.text_fields) > 1 else self.text_fields[0]
        )

        if is_batched(self.modify_document):
            dataset.df[text_fields] = dataset.df[text_fields].map_partitions(
                self.modify_document, meta=self.meta
            )
        else:
            dataset.df[text_fields] = dataset.df[text_fields].apply(
                self.modify_document, meta=self.meta
            )

        return dataset
