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
from typing import List, Optional, Union

import dask.dataframe as dd
from fsspec.core import open_files


class ImageTextPairDataset:
    def __init__(self, path: str, metadata, tar_files: List[str]) -> None:
        self.path = path
        self.metadata = metadata
        self.tar_files = tar_files

    @classmethod
    def from_webdataset(cls, path: str):
        metadata = dd.read_parquet(path)
        tar_files = cls._get_tar_files(path)

        return cls(path, metadata, tar_files)

    @staticmethod
    def _get_tar_files(path: str) -> List[str]:
        glob_str = os.path.join(path, "*.tar")
        # open_files doesn't actually open a file descriptor
        tar_files = [file.path for file in open_files(glob_str)]

        return tar_files

    def save_metadata(
        self, path: Optional[str] = None, columns: Optional[List[str]] = None
    ) -> None:
        if path is None:
            path = self.path

        if columns is None:
            metadata = self.metadata
        else:
            metadata = self.metadata[columns]

        metadata.to_parquet(path)

    def reshard(self, path: str, filter_column: str) -> None:
        pass
