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

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cpu", action="store_true", default=False, help="Run tests without gpu marker"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="Skipping GPU tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
