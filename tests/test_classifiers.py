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

import pytest
from distributed import Client

from nemo_curator import get_client
from nemo_curator.classifiers import DomainClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


@pytest.fixture
def gpu_client(request):
    with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
        request.client = client
        request.cluster = cluster
        yield


@pytest.fixture
def domain_dataset():
    text = [
        "Quantum computing is set to revolutionize the field of cryptography.",
        "Investing in index funds is a popular strategy for long-term financial growth.",
        "Recent advancements in gene therapy offer new hope for treating genetic disorders.",
        "Online learning platforms have transformed the way students access educational resources.",
        "Traveling to Europe during the off-season can be a more budget-friendly option.",
    ]
    df = cudf.DataFrame({"text": text})
    df = dask_cudf.from_cudf(df, 1)
    return DocumentDataset(df)


@pytest.mark.gpu
def test_domain_classifier(gpu_client, domain_dataset):
    classifier = DomainClassifier()
    result_dataset = classifier(dataset=domain_dataset)
    result_pred = result_dataset.df.compute()["domain_pred"]

    expected_pred = cudf.Series(
        [
            "Computers_and_Electronics",
            "Finance",
            "Health",
            "Jobs_and_Education",
            "Travel_and_Transportation",
        ]
    )

    assert result_pred.equals(expected_pred)
