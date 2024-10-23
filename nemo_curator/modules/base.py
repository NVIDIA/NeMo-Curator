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
    SUPPORTED_BACKENDS = ["pandas", "cudf", "any"]

    def __init__(self, input_backend: str, name=None) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__

        if input_backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"{input_backend} not one of the supported backends {self.SUPPORTED_BACKENDS}"
            )
        self.input_backend = input_backend

    @abstractmethod
    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        raise NotImplementedError("call method must be implemented by subclasses")

    def _check_backend(self, partition, partition_info=None):
        if partition_info is None:
            return

        backend = type(partition).__module__.split(".")[0]
        if backend != self.input_backend:
            raise ValueError(
                f"Module {self.name} requires dataset to have backend {self.input_backend} but got backend {backend}"
            )

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        if self.input_backend != "any":
            dataset.df.map_partitions(self._check_backend)

        return self.call(dataset)
