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

from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


# Global variable to store the cache directory
_global_cache_dir = None

def initialize_cache_directory(cache_dir: str) -> str:
    """
    Initialize and set the global cache directory.
    """
    global _global_cache_dir
    cache_dir = expand_outdir_and_mkdir(cache_dir)
    _global_cache_dir = cache_dir
    return cache_dir


def get_cache_directory() -> str:
    """
    Retrieve the global cache directory.
    """
    return _global_cache_dir
