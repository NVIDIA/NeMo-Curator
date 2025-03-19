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
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dask.dataframe.utils import assert_eq
from transformers import AutoConfig, AutoModel, AutoTokenizer
import tempfile
import pandas as pd
import cupy as cp

from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from
from nemo_curator.modules.semantic_dedup.clusteringmodel import add_l2_dist_to_cents
from nemo_curator.utils.semdedup_utils import rank_within_cluster, get_normalized_embedding_array

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
EmbeddingCreator = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.embeddings", "EmbeddingCreator"
)
pairwise_cosine_similarity = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity"
)
pairwise_cosine_similarity_batched = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity_batched"
)
if TYPE_CHECKING:
    from nemo_curator.utils.semdedup_utils import (
        pairwise_cosine_similarity,
        pairwise_cosine_similarity_batched,
    )


@pytest.fixture
def dedup_data():
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
    return DocumentDataset(df)


@pytest.fixture
def non_dedup_data():
    df = cudf.DataFrame(
        {
            "doc_id": ["doc_1", "doc_2"],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "A test string",
            ],
        }
    )
    df = dask_cudf.from_cudf(df, 2)
    return DocumentDataset(df)


@pytest.mark.gpu
class TestSemDuplicates:
    @pytest.mark.parametrize("n_clusters", [3, 10])
    def test_sem_dedup(
        self,
        dedup_data,
        tmpdir,
        n_clusters,
        gpu_client,
    ):
        print("client", gpu_client)

        cache_dir = os.path.join(tmpdir, "test_sem_dedup_cache")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="id",
            id_column_type="int",
        )

        dedup_data_len = dedup_data.df.shape[0].compute()
        if n_clusters > dedup_data_len:
            # Number of records in the dataset should never be less than n_clusters
            with pytest.raises(ValueError):
                result = sem_duplicates(dedup_data)
        else:
            # Correctly returns the original dataset with no duplicates removed
            result = sem_duplicates(dedup_data)
            result_df = result.df.compute()
            duplicate_docs = [2, 3, 4, 200, 300]
            expected_df = cudf.Series(duplicate_docs, name="id")
            assert_eq(result_df["id"].sort_values(), expected_df, check_index=False)

    @pytest.mark.parametrize("n_clusters", [2, 3])
    def test_no_sem_dedup(
        self,
        non_dedup_data,
        tmpdir,
        n_clusters,
        gpu_client,
    ):
        print("client", gpu_client)

        cache_dir = os.path.join(tmpdir, "test_no_sem_dedup")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="doc_id",
            id_column_type="str",
        )

        non_dedup_data_len = non_dedup_data.df.shape[0].compute()
        if n_clusters > non_dedup_data_len:
            # Number of records in the dataset should never be less than n_clusters
            with pytest.raises(ValueError):
                result = sem_duplicates(non_dedup_data)
        else:
            # Correctly returns the original dataset with no duplicates removed
            result = sem_duplicates(non_dedup_data)
            result_df = result.df.compute()
            duplicate_docs = ["doc_1", "doc_2"]
            expected_df = cudf.Series(duplicate_docs, name="doc_id")
            assert_eq(result_df["doc_id"].sort_values(), expected_df, check_index=False)

    @pytest.mark.parametrize("pooling_strategy", ["last_token", "mean_pooling"])
    def test_embedding_creator_pooling_strategies(self, tmpdir, pooling_strategy):
        test_text_1 = "The quick brown fox jumps over the lazy dog"
        test_text_2 = "The brown fox jumps over the dog"
        test_texts = [test_text_1, test_text_2] * 32
        df = cudf.DataFrame({"text": test_texts})
        ddf = dask_cudf.from_cudf(df, 1)

        cache_dir = os.path.join(tmpdir, "test_embeddings_cache")

        embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            embedding_batch_size=32,
            embedding_pooling_strategy=pooling_strategy,
            input_column="text",
            embedding_output_dir=os.path.join(cache_dir, "mean_embeddings"),
        )

        embeddings = embedding_creator.create_embeddings(ddf).compute()
        embeddings = embeddings["embeddings"].to_arrow().to_pylist()
        embeddings = np.array(embeddings)

        reference_embeddings = get_reference_embeddings(
            test_texts, pooling_strategy=pooling_strategy
        )

        assert np.allclose(
            embeddings, reference_embeddings, atol=1e-3
        ), "Embeddings should match reference embeddings"


def get_reference_embeddings(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pooling_strategy="last_token",
):
    """
    Get embeddings using either last token or mean pooling strategy.

    Args:
        texts: List of input texts
        model_name: Name or path of the model to use
        pooling_strategy: Either "last_token" for last token or "mean" for mean pooling
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda")
    model.eval()
    max_len_to_use = tokenizer.model_max_length
    if max_len_to_use > 1e5:
        max_len_to_use = AutoConfig.from_pretrained(model_name).max_position_embeddings
    max_seq_length: int = max_len_to_use

    embs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs = model(**inputs)

        if pooling_strategy == "last_token":
            embeddings = outputs.last_hidden_state[:, -1, :]
        elif pooling_strategy == "mean_pooling":
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError(
                "pooling_strategy must be either 'last_token' or 'mean_pooling'"
            )

        normed_emb = F.normalize(embeddings, dim=1).cpu()
        normed_emb = normed_emb.squeeze(0)
        embs.append(normed_emb)

    return np.array(embs)


class TestSemDedupUtils:
    def setup_method(self):
        # We create a 6x3 array where each row is a unit vector
        # The second and last two rows are the same
        input_embeddings = torch.tensor(
            np.asarray(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [1, 2, 3], [1, 2, 3]],
            ),
            dtype=torch.float32,
        )
        # Normalize the input array
        self.input_embeddings = input_embeddings / torch.norm(
            input_embeddings, dim=1, keepdim=True
        )
        self.expected_pairwise_similarity = torch.tensor(
            [0.0000, 0.974631, 0.998190, 0.999618, 1.0000, 1.0000]
        )
        self.expected_indices = [0, 0, 1, 2, 0, 0]

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_pairwise_cosine_similarity(self, device: Literal["cpu", "cuda"]):
        max_similarity, max_indices = pairwise_cosine_similarity(
            self.input_embeddings.to(device), device
        )
        torch.testing.assert_close(
            max_similarity, self.expected_pairwise_similarity, rtol=1e-6, atol=1e-6
        )
        assert max_indices == self.expected_indices

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6])
    def test_pairwise_cosine_similarity_batched(
        self, device: Literal["cpu", "cuda"], batch_size: int
    ):
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            self.input_embeddings.to(device), device, batch_size
        )
        torch.testing.assert_close(max_similarity, self.expected_pairwise_similarity)
        assert max_indices == self.expected_indices

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    @pytest.mark.parametrize("batch_size", [100, 512, 1024, 2048])
    def test_pairwise_cosine_similarity_batched_rand_array(
        self, device: Literal["cpu", "cuda"], batch_size: int
    ):
        N = 1024
        D = 512
        rand_arr = torch.randn(N, D, device=device)
        max_similarity, max_indices = pairwise_cosine_similarity(rand_arr, device)
        max_similarity_batched, max_indices_batched = (
            pairwise_cosine_similarity_batched(rand_arr, device, batch_size=batch_size)
        )
        torch.testing.assert_close(
            max_similarity, max_similarity_batched, rtol=1e-5, atol=1e-5
        )
        assert max_indices == max_indices_batched

    @pytest.mark.gpu
    @pytest.mark.parametrize("keep_hard", [True, False])
    def test_rank_within_cluster(self, keep_hard: bool):
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock data setup
            cluster_id = 0
            id_col = "id"
            embedding_col = "embedding"
            nearest_cent_dir = temp_dir
            output_sorted_clusters_dir = temp_dir

            # Create mock centroids and cluster data
            centroids = self.input_embeddings[:1]
            cluster_data = cudf.DataFrame(
                {
                    id_col: list(range(self.input_embeddings.shape[0])),
                    embedding_col: self.input_embeddings.tolist(),
                }
            )

            # Save mock cluster data to a file
            cluster_data_path = os.path.join(
                nearest_cent_dir, f"nearest_cent={cluster_id}"
            )
            cluster_data.to_parquet(cluster_data_path)

            # Call the function
            rank_within_cluster(
                id_col=id_col,
                nearest_cent_dir=nearest_cent_dir,
                output_sorted_clusters_dir=output_sorted_clusters_dir,
                centroids=centroids,
                embedding_col=embedding_col,
                sim_metric="cosine",
                keep_hard=keep_hard,
                cluster_ids=[cluster_id],
            )

            # Load the sorted cluster
            sorted_cluster_file_path = os.path.join(
                output_sorted_clusters_dir, f"cluster_{cluster_id}.npy"
            )
            sorted_cluster = np.load(sorted_cluster_file_path)

            # Expected order based on cosine similarity
            if keep_hard:
                expected_order = [
                    # id, dist_to_cent, cluster_id
                    [3, 0.9513, 0],
                    [2, 0.9594, 0],
                    [1, 0.9746, 0],
                    [0, 1.0, 0],
                    [4, 1.0000, 0],
                    [5, 1.0000, 0],
                ]
            else:
                expected_order = [
                    # id, dist_to_cent, cluster_id
                    [0, 1.0, 0],
                    [4, 1.0000, 0],
                    [5, 1.0000, 0],
                    [1, 0.9746, 0],
                    [2, 0.9594, 0],
                    [3, 0.9513, 0],
                ]
            expected_order = np.array(expected_order)
            expected_order[:, 1] = 1 - expected_order[:, 1]
            pd.testing.assert_frame_equal(
                pd.DataFrame(sorted_cluster),
                pd.DataFrame(expected_order),
                check_exact=False,
                rtol=1e-3,
                atol=1e-3,
            )

    @pytest.mark.gpu
    def test_get_normalized_embedding_array(self):
        # Mock data setup
        df = cudf.DataFrame({
            "embedding": [[3, 4, 5], [1, 2, 2], [1, 0, 0]],
        })
        expected_normalized = cp.array([
            [0.42426407, 0.565685, 0.707107],
            [0.33333334, 0.6666667, 0.6666667],
            [1.0, 0.0, 0.0],
        ])

        # Call the function
        normalized_embeddings = get_normalized_embedding_array(df, "embedding")

        # Assert the normalized embeddings match the expected values
        cp.testing.assert_allclose(normalized_embeddings, expected_normalized, rtol=1e-5, atol=1e-5)


    @pytest.mark.gpu
    def test_add_l2_dist_to_cents(self):
        # Mock data setup
        df = cudf.DataFrame({
            "nearest_cent": [0, 1, 0],
            # Here 1,1 is not normalized therefore it'll get normalized to 0.707107, 0.707107
            "embedding": [[1, 0], [0, 1], [1, 1]],
        })
        centroids = cp.array([[1, 0], [0, 1]])
        # The distance between [0.707107, 0.707107] and [1, 0] is 
        # [(0.707 - 1) ** 2 + (0.707 - 0) ** 2]**0.5 = sqrt(0.085 + 0.499) = sqrt(0.585) = 0.7653
        expected_distances = cp.array([0.0, 0.0,  0.76530])
        # Call the function
        df_with_distances = add_l2_dist_to_cents(df, "embedding", centroids)

        # Assert the distances match the expected values
        np.testing.assert_almost_equal(
            df_with_distances["l2_dist_to_cent"].to_arrow().to_pylist(), 
            expected_distances.tolist(), 
            decimal=4
        )
            