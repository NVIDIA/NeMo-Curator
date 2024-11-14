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

import dask.dataframe as dd
import pandas as pd

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset


def all_equal(left_result: pd.DataFrame, right_result: pd.DataFrame):
    l_cols = set(left_result.columns)
    r_cols = set(right_result.columns)
    assert l_cols == r_cols
    for col in left_result.columns:
        left = left_result[col].reset_index(drop=True)
        right = right_result[col].reset_index(drop=True)
        assert all(left == right), f"Mismatch in {col} column.\n{left}\n{right}\n"


class TestDocumentDataset:
    def test_to_from_pandas(self):
        original_df = pd.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        dataset = DocumentDataset.from_pandas(original_df)
        converted_df = dataset.to_pandas()
        all_equal(original_df, converted_df)
