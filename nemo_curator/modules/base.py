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

from nemo_curator.datasets import DocumentDataset


class Module(ABC):
    def __init__(self, name=None) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__

    @abstractmethod
    @property
    def input_backend(self) -> str:
        raise NotImplementedError(
            "input_backend method must be implemented by subclasses"
        )

    @abstractmethod
    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        raise NotImplementedError("call method must be implemented by subclasses")

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        if self.input_backend != "any" and dataset.df.backend != self.input_backend:
            raise ValueError(
                f"Module {self.name} requires dataset to have backend {self.input_backend} but got backend {dataset.df.backend}"
            )

        return self.call(dataset)
