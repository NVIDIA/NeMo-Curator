# Copyright (c) 2024, NVIDIA CORPORATION.
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

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.file_utils import get_all_files_paths_under


def ensure_directory_exists(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def process_data(input_folder, output_folder, prefix, partition_size="512MB"):
    raw_files = get_all_files_paths_under(input_folder)
    raw_data = DocumentDataset.read_parquet(raw_files)
    raw_data_rep = DocumentDataset(
        raw_data.df.repartition(partition_size=partition_size)
    )
    add_id = AddId(id_field="nemo_id", id_prefix=prefix)
    data_with_id = add_id(raw_data_rep)
    ensure_directory_exists(output_folder)
    data_with_id.to_parquet(output_folder)
