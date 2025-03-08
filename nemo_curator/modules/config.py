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

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from nemo_curator.cache import Cache


@dataclass
class BaseConfig:
    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            yaml_dict = yaml.safe_load(file)
        return cls(**yaml_dict)


@dataclass
class FuzzyDuplicatesConfig(BaseConfig):
    """
    Configuration for MinHash-based fuzzy duplicates detection.

    Parameters
    ----------
    cache_dir: If specified, directory to store deduplication intermediates, such as
        minhashes, buckets, etc. If None, we check if a cache_dir has been initialized
        with Cache().get_cache_directory(). Default is None.
    profile_dir: If specified, directory to write Dask profile. Default is None.
    id_field: Column in the dataset denoting document ID. Default is "id".
    text_field: Column in the dataset denoting document content. Default is "text".
    perform_removal: Boolean value to specify whether calling the module should remove
        the duplicates from the original dataset, or return the list of IDs denoating
        duplicates. Default is False.
    seed: Seed for minhash permutations. Default is 42.
    char_ngrams: Size of character n-gram shingles used in minhash computation.
        Default is 5.
    num_buckets: Number of bands or buckets to use during Locality Sensitive Hashing.
        Default is 20.
    hashes_per_bucket: Number of hashes per bucket/band. Default is 13.
    use_64_bit_hash: Whether to use a 32bit or 64bit hash function for minhashing.
        Default is False.
    buckets_per_shuffle: Number of bands/buckets to shuffle concurrently. Larger values
        process larger batches by processing multiple bands but might lead to memory
        pressures and related errors. Default is 1.
    false_positive_check: Whether to run a check to look for false positives within
        buckets. Note: This is a computationally expensive step. Default is False.
    num_anchors: Number of documents per bucket to use as reference for computing
        Jaccard pairs within that bucket to identify false positives. Default is 2.
    jaccard_threshold: The Jaccard similariy threshold to consider a document a near
        duplicate during false positive evaluations. Default is 0.8.
    bucket_mapping_blocksize: Default is 256.
    parts_per_worker: Default is 1.
    bucket_parts_per_worker: Default is 8.
    """

    # General config
    cache_dir: Optional[str] = None
    profile_dir: Optional[str] = None
    id_field: str = "id"
    text_field: str = "text"
    perform_removal: bool = False

    # Minhash + LSH config
    seed: int = 42
    char_ngrams: int = 24
    num_buckets: int = 20
    hashes_per_bucket: int = 13
    use_64_bit_hash: bool = False
    buckets_per_shuffle: int = 1

    false_positive_check: bool = False
    # Only required for false positive check
    num_anchors: Optional[int] = None
    jaccard_threshold: Optional[float] = None
    bucket_mapping_blocksize: Optional[int] = None
    parts_per_worker: Optional[int] = None
    bucket_parts_per_worker: Optional[int] = None

    def __post_init__(self):
        self.num_hashes = self.num_buckets * self.hashes_per_bucket

        false_positive_defaults = {
            "num_anchors": 2,
            "jaccard_threshold": 0.8,
            "bucket_mapping_blocksize": 256,
            "parts_per_worker": 1,
            "bucket_parts_per_worker": 8,
        }

        if self.false_positive_check:
            warnings.warn(
                "Identifying false positives during Minhash deduplication is "
                "computationally expensive. For improved performance consider setting "
                "this to False."
            )

            for arg, default in false_positive_defaults.items():
                if getattr(self, arg) is None:
                    setattr(self, arg, default)

            if self.num_anchors <= 0:
                raise ValueError("Number of anchors must be greater than 0.")

            if self.num_anchors > 2:
                warnings.warn(
                    "Using a higher number of anchor documents might lead to higher memory "
                    "footprint and might impact performance.",
                    category=UserWarning,
                )

            if not 0 <= self.jaccard_threshold <= 1:
                raise ValueError("Jaccard threshold must be between [0, 1].")

        else:
            if self.char_ngrams < 20:
                warnings.warn(
                    "Using a small char_ngrams value might lead to a large number (~5%) of false positives during deduplication."
                    " Using a value of at least 20 for char_ngrams is recommended."
                )

            unused_false_positive_args = [
                arg
                for arg in false_positive_defaults.keys()
                if getattr(self, arg) is not None
            ]

            if unused_false_positive_args:
                warnings.warn(
                    f"False positive check is disabled. Unused arguments {unused_false_positive_args} will be ignored.",
                    category=UserWarning,
                )

        if not 1 <= self.buckets_per_shuffle <= self.num_buckets:
            raise ValueError("Buckets per shuffle must be between [1, num_buckets].")

        if self.cache_dir is None:
            cache_dir = Cache().get_cache_directory()
            if cache_dir is None:
                raise ValueError(
                    "Finding fuzzy duplicates requires a cache directory accessible via "
                    "all workers to store intermediates. Please use "
                    "Cache(cache_dir=...) or FuzzyDuplicatesConfig(cache_dir=...) to "
                    "set the cache directory."
                )
            else:
                self.cache_dir = cache_dir

        if not self.perform_removal:
            warnings.warn(
                "In future NeMo Curator releases, the default value for perform_removal will be True."
            )


@dataclass
class SemDedupConfig(BaseConfig):
    """
    Configuration for Semantic Deduplication.

    Attributes:
        cache_dir (Optional[str]): If specified, directory to store cache.
            If None, we check if a cache_dir has been initialized with Cache().get_cache_directory().
            Default is None.
        profile_dir (Optional[str]): If specified, directory to write Dask profile.
            Default is None.
        num_files (int): Number of files. Default is -1, meaning all files.

        embedding_model_name_or_path (str): Model name or path for embeddings.
            Default is "sentence-transformers/all-MiniLM-L6-v2".
        embedding_batch_size (int): Initial batch size for processing embeddings.
            Default is 128.
        embeddings_save_loc (str): Location to save embeddings.
            Default is "embeddings".
        embedding_max_mem_gb (int): Maximum memory usage in GB for the embedding process.
            If None, it defaults to the available GPU memory minus 4 GB.
        embedding_pooling_strategy (str): Strategy for pooling embeddings, either
            "mean_pooling" or "last_token". Default is "mean_pooling".
        embedding_column (str): The column name that stores the embeddings.
            Default is "embeddings".
        write_embeddings_to_disk (bool): If True, saves the embeddings to disk.
            We recommend setting this to False when you have a delayed pipeline.
            Setting it to False can lead to more memory overhead. Default is True.
        write_to_filename (bool): If True, saves the embeddings to the same filename as input files.
            Default False.

        max_iter (int): Maximum iterations for clustering. Default is 100.
        n_clusters (int): Number of clusters. Default is 1000.
        clustering_save_loc (str): Location to save clustering results.
            Default is "clustering_results".
        random_state (int): KMeans random state used for reproducibility. Default is 1234.
        sim_metric (str): Similarity metric for deduplication.
            Default is "cosine".
        which_to_keep (str): Method to determine which duplicates to keep.
            Default is "hard".
        sort_clusters (bool): Whether to sort clusters. Default is True.
        kmeans_with_cos_dist (bool): Whether or not to use KMeans with cosine distance.
            Default is False.
        clustering_input_partition_size (str): The size of data partition with which to run KMeans.
            Default is "2gb".

        eps_thresholds (List[float]): Epsilon thresholds to calculate if semantically
            similar or not. Default is [0.01, 0.001].
        eps_to_extract (float): Epsilon value to extract deduplicated data.
            Default is 0.01.
    """

    cache_dir: str = None
    profile_dir: Optional[str] = None
    num_files: int = -1

    # Embeddings
    embedding_model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    embeddings_save_loc: str = "embeddings"
    embedding_max_mem_gb: Optional[int] = None
    # Options: "mean_pooling", "last_token"
    embedding_pooling_strategy: str = "mean_pooling"
    embedding_column: str = "embeddings"
    write_embeddings_to_disk: bool = True
    write_to_filename: bool = False

    # Clustering
    max_iter: int = 100
    n_clusters: int = 1000
    clustering_save_loc: str = "clustering_results"
    random_state: int = 1234
    sim_metric: str = "cosine"
    which_to_keep: str = "hard"
    sort_clusters: bool = True
    kmeans_with_cos_dist: bool = False
    clustering_input_partition_size: str = "2gb"

    # SemDedup
    eps_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.001])
    eps_to_extract: float = 0.01

    def __post_init__(self):
        if self.cache_dir is None:
            cache_dir = Cache().get_cache_directory()
            if cache_dir is None:
                raise ValueError(
                    "Finding semantic duplicates requires a cache directory accessible "
                    "via all workers to store intermediates. Please use "
                    "Cache(cache_dir=...) or SemDedupConfig(cache_dir=...) to "
                    "set the cache directory."
                )
            else:
                self.cache_dir = cache_dir

        if self.eps_to_extract not in self.eps_thresholds:
            raise ValueError(
                f"Epsilon to extract {self.eps_to_extract} must be in eps_thresholds "
                f"{self.eps_thresholds}."
            )
