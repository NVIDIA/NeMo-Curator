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

from nemo_curator.datasets.doc_dataset import DocumentDataset
from nemo_curator.modules.base import Module


class ToBackend(Module):
    def __init__(self, backend: str) -> None:
        super().__init__(input_backend="any")
        self.backend = backend

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        return DocumentDataset(dataset.df.to_backend(self.backend))
