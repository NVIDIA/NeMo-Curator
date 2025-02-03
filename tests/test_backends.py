# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from dask.dataframe.utils import assert_eq
from distributed import Client

from nemo_curator import Module
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


class CPUModule(Module):
    def __init__(self):
        super().__init__(input_backend="pandas")

    def call(dataset: DocumentDataset):
        dataset.df["cpu_length"] = dataset.df["text"].str.len()
        return dataset


class GPUModule(Module):
    def __init__(self):
        super().__init__(input_backend="cudf")

    def call(dataset: DocumentDataset):
        dataset.df["gpu_length"] = dataset.df["text"].str.len()
        return dataset


@pytest.fixture
def cpu_data():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 100, 200, 300],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "The quick brown foxes jumps over the lazy dog",
                "The quick brown wolf jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
                "A test string",
                "Another test string",
                "A different object",
            ],
        }
    )
    gt_lengths = pd.Series([43, 45, 44, 43, 13, 19, 18], name="cpu_lengths")
    return DocumentDataset.from_pandas(df), gt_lengths


@pytest.fixture
def gpu_data():
    df = cudf.DataFrame(
        {
            "id": [1, 2, 3, 4, 100, 200, 300],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "The quick brown foxes jumps over the lazy dog",
                "The quick brown wolf jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
                "A test string",
                "Another test string",
                "A different object",
            ],
        }
    )
    df = dask_cudf.from_cudf(df, 2)
    gt_lengths = cudf.Series([43, 45, 44, 43, 13, 19, 18], name="gpu_lengths")
    return DocumentDataset(df), gt_lengths


@pytest.mark.gpu
class TestBackendSupport:
    @pytest.fixture(autouse=True, scope="class")
    def gpu_client(self, request):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.cls.client = client
            request.cls.cluster = cluster
            yield

    def test_pandas_backend(
        self,
        cpu_data,
    ):
        print("client", self.client)
        dataset, gt_lengths = cpu_data
        pipeline = CPUModule()
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["cpu_lengths"], gt_lengths)
