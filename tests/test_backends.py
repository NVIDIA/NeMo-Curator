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
import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq
from distributed import Client

from nemo_curator import Module, Sequential, ToBackend
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


class CPUModule(Module):
    def __init__(self):
        super().__init__(input_backend="pandas")

    def call(self, dataset: DocumentDataset):
        dataset.df["cpu_lengths"] = dataset.df["text"].str.len()
        return dataset


class GPUModule(Module):
    def __init__(self):
        super().__init__(input_backend="cudf")

    def call(self, dataset: DocumentDataset):
        dataset.df["gpu_lengths"] = dataset.df["text"].str.len()
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
    gt_lengths = cudf.Series(
        [43, 45, 44, 43, 13, 19, 18], name="gpu_lengths", dtype="int32"
    )
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

    def test_cudf_backend(
        self,
        gpu_data,
    ):
        print("client", self.client)
        dataset, gt_lengths = gpu_data
        pipeline = GPUModule()
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["gpu_lengths"], gt_lengths)

    def test_pandas_to_cudf(
        self,
        cpu_data,
        gpu_data,
    ):
        print("client", self.client)
        dataset, gt_cpu_lengths = cpu_data
        _, gt_gpu_lengths = gpu_data
        pipeline = Sequential(
            [
                CPUModule(),
                ToBackend("cudf"),
                GPUModule(),
            ]
        )
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["cpu_lengths"], gt_cpu_lengths)
        assert_eq(result_df["gpu_lengths"], gt_gpu_lengths)

    def test_cudf_to_pandas(
        self,
        cpu_data,
        gpu_data,
    ):
        print("client", self.client)
        _, gt_cpu_lengths = cpu_data
        dataset, gt_gpu_lengths = gpu_data
        pipeline = Sequential(
            [
                GPUModule(),
                ToBackend("pandas"),
                CPUModule(),
            ]
        )
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["cpu_lengths"], gt_cpu_lengths)
        assert_eq(result_df["gpu_lengths"], gt_gpu_lengths)

    def test_5x_switch(
        self,
        cpu_data,
        gpu_data,
    ):
        print("client", self.client)
        dataset, gt_cpu_lengths = cpu_data
        _, gt_gpu_lengths = gpu_data
        pipeline = Sequential(
            [
                CPUModule(),
                ToBackend("cudf"),
                GPUModule(),
                ToBackend("pandas"),
                CPUModule(),
                ToBackend("cudf"),
                GPUModule(),
                ToBackend("pandas"),
                CPUModule(),
                ToBackend("cudf"),
                GPUModule(),
            ]
        )
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["cpu_lengths"], gt_cpu_lengths)
        assert_eq(result_df["gpu_lengths"], gt_gpu_lengths)
