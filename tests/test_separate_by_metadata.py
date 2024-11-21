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

import pandas as pd
import pytest
from dask import dataframe as dd
from dask.dataframe.utils import assert_eq

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import write_to_disk
from nemo_curator.utils.file_utils import separate_by_metadata


@pytest.fixture
def tmp_path_w_data(tmp_path):
    def _write_data(num_files, file_ext):
        out_path = tmp_path / file_ext
        df = pd.DataFrame(
            {
                "id": [1, 2, 300, 4, -1],
                "text": ["abc", "aba", "abb", "aba", "abc"],
                "metadata": ["doc", "code", "test", "code", "doc"],
            }
        )
        dfs = []
        for i in range(num_files):
            partition = df.copy()
            partition["filename"] = f"f{i}.{file_ext}"
            dfs.append(partition)

        df = dd.concat(dfs)
        write_to_disk(
            df=df,
            output_file_dir=str(out_path),
            write_to_filename=True,
            output_type=file_ext,
        )
        return out_path

    return _write_data


@pytest.mark.parametrize(
    "backend", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
)
class TestMetadataSep:

    @pytest.mark.parametrize("files_per_partition", [1, 3])
    @pytest.mark.parametrize(
        "file_ext, read_f",
        [
            ("jsonl", DocumentDataset.read_json),
            ("parquet", DocumentDataset.read_parquet),
        ],
    )
    def test_metadatasep(
        self, tmp_path_w_data, files_per_partition, backend, file_ext, read_f
    ):
        data_dir = tmp_path_w_data(num_files=5, file_ext=file_ext)
        output_dir = data_dir / "metadata_sep"
        df = read_f(
            str(data_dir),
            backend=backend,
            files_per_partition=files_per_partition,
            add_filename=True,
        ).df
        separate_by_metadata(
            input_data=df,
            output_dir=str(output_dir),
            metadata_field="metadata",
            output_type=file_ext,
        ).compute()

        found_metadata = set(os.listdir(output_dir))
        expected_metadata = {"code", "doc", "test"}
        assert found_metadata == expected_metadata

        dfs = []
        for metadata in expected_metadata:
            meta_df = read_f(
                str(output_dir / metadata),
                backend=backend,
                files_per_partition=1,
                add_filename=True,
            ).df
            dfs.append(meta_df)
        got_df = dd.concat(dfs)
        assert_eq(got_df, df, check_index=False)
