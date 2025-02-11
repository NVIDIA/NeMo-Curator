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

from nemo_curator import (
    BaseModule,
    FuzzyDuplicates,
    FuzzyDuplicatesConfig,
    ScoreFilter,
    Sequential,
    ToBackend,
)
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import MeanWordLengthFilter
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


class CPUModule(BaseModule):
    def __init__(self):
        super().__init__(input_backend="pandas")

    def call(self, dataset: DocumentDataset):
        dataset.df["cpu_lengths"] = dataset.df["text"].str.len()
        return dataset


class GPUModule(BaseModule):
    def __init__(self):
        super().__init__(input_backend="cudf")

    def call(self, dataset: DocumentDataset):
        dataset.df["gpu_lengths"] = dataset.df["text"].str.len()
        return dataset


class AnyModule(BaseModule):
    def __init__(self):
        super().__init__(input_backend="any")

    def call(self, dataset: DocumentDataset):
        dataset.df["any_lengths"] = dataset.df["text"].str.len()
        return dataset


@pytest.fixture
def raw_data():
    base_data = {
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
    gt_results = [43, 45, 44, 43, 13, 19, 18]

    return base_data, gt_results


@pytest.fixture
def cpu_data(raw_data):
    base_data, gt_results = raw_data
    df = pd.DataFrame(base_data)
    gt_lengths = pd.Series(gt_results, name="cpu_lengths")
    return DocumentDataset.from_pandas(df), gt_lengths


@pytest.fixture
def gpu_data(raw_data):
    base_data, gt_results = raw_data
    df = cudf.DataFrame(base_data)
    df = dask_cudf.from_cudf(df, 2)
    gt_lengths = cudf.Series(gt_results, name="gpu_lengths", dtype="int32")
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

    def test_any_backend(
        self,
        cpu_data,
        gpu_data,
    ):
        print("client", self.client)
        cpu_dataset, gt_cpu_lengths = cpu_data
        gt_cpu_lengths = gt_cpu_lengths.rename("any_lengths")
        gpu_dataset, gt_gpu_lengths = gpu_data
        gt_gpu_lengths = gt_gpu_lengths.rename("any_lengths")
        pipeline = AnyModule()

        cpu_result = pipeline(cpu_dataset)
        cpu_result_df = cpu_result.df.compute()
        assert_eq(cpu_result_df["any_lengths"], gt_cpu_lengths)
        gpu_result = pipeline(gpu_dataset)
        gpu_result_df = gpu_result.df.compute()
        assert_eq(gpu_result_df["any_lengths"], gt_gpu_lengths)

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
        assert sorted(list(result_df.columns)) == [
            "cpu_lengths",
            "gpu_lengths",
            "id",
            "text",
        ]
        assert_eq(result_df["cpu_lengths"], gt_cpu_lengths)
        assert_eq(result_df["gpu_lengths"], gt_gpu_lengths)

    def test_wrong_backend_cpu_data(self, cpu_data):
        with pytest.raises(ValueError):
            print("client", self.client)
            dataset, _ = cpu_data
            pipeline = GPUModule()
            result = pipeline(dataset)
            _ = result.df.compute()

    def test_wrong_backend_gpu_data(self, gpu_data):
        with pytest.raises(ValueError):
            print("client", self.client)
            dataset, _ = gpu_data
            pipeline = CPUModule()
            result = pipeline(dataset)
            _ = result.df.compute()

    def test_unsupported_to_backend(self, cpu_data):
        with pytest.raises(ValueError):
            print("client", self.client)
            dataset, _ = cpu_data
            pipeline = ToBackend("fake_backend")
            result = pipeline(dataset)
            _ = result.df.compute()


@pytest.fixture
def real_module_raw_data():
    base_data = {
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
    return base_data


@pytest.fixture
def real_module_cpu_data(real_module_raw_data):
    df = pd.DataFrame(real_module_raw_data)
    gt_results = pd.Series(
        [35 / 9, 37 / 9, 4.0, 35 / 9, 33 / 9, 51 / 9, 48 / 9], name="mean_lengths"
    )
    return DocumentDataset.from_pandas(df), gt_results


@pytest.fixture
def real_module_gpu_data(real_module_raw_data):
    df = cudf.DataFrame(real_module_raw_data)
    df = dask_cudf.from_cudf(df, 2)
    gt_results = cudf.Series([[1, 2, 3, 4], [100, 200]], name="id")
    return DocumentDataset(df), gt_results


@pytest.mark.gpu
class TestRealModules:
    @pytest.fixture(autouse=True, scope="class")
    def gpu_client(self, request):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.cls.client = client
            request.cls.cluster = cluster
            yield

    def test_score_filter(
        self,
        real_module_cpu_data,
    ):
        print("client", self.client)
        dataset, gt_results = real_module_cpu_data
        pipeline = ScoreFilter(
            MeanWordLengthFilter(), score_field="mean_lengths", score_type=float
        )
        result = pipeline(dataset)
        result_df = result.df.compute()
        assert_eq(result_df["mean_lengths"], gt_results)

    def test_score_filter_wrong_backend(
        self,
        real_module_gpu_data,
    ):
        with pytest.raises(ValueError):
            print("client", self.client)
            dataset, _ = real_module_gpu_data
            pipeline = ScoreFilter(
                MeanWordLengthFilter(), score_field="mean_lengths", score_type=float
            )
            result = pipeline(dataset)
            _ = result.df.compute()
