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

from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


class Cache:
    _instance = None
    _cache_dir = None

    def __new__(cls, cache_dir=None):
        if cls._instance is None:
            cls._instance = super(Cache, cls).__new__(cls)
            if cache_dir is not None:
                cls._cache_dir = expand_outdir_and_mkdir(cache_dir)
            else:
                cls._cache_dir = None
        elif cache_dir is not None and cls._cache_dir is None:
            cls._cache_dir = expand_outdir_and_mkdir(cache_dir)
        return cls._instance

    @classmethod
    def get_cache_directory(cls) -> str:
        """
        Retrieve the cache directory.
        """
        return cls._cache_dir
