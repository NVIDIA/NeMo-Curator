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
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dask.dataframe.utils import assert_eq
from distributed import Client
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.semantic_dedup.embeddings import EmbeddingCreator
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


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


@pytest.mark.gpu
class TestSemDuplicates:
    @pytest.fixture(autouse=True, scope="class")
    def gpu_client(self, request):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.cls.client = client
            request.cls.cluster = cluster
            yield

    def test_sem_dedup(
        self,
        dedup_data,
        tmpdir,
    ):
        print("client", self.client)
        cache_dir = os.path.join(tmpdir, "test_sem_dedup_cache")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            seed=42,
            n_clusters=3,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )
        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="id",
            id_column_type="int",
        )
        result = sem_duplicates(dedup_data)
        result_df = result.df.compute()
        duplicate_docs = [2, 3, 4, 200, 300]
        expected_df = cudf.Series(duplicate_docs, name="id")
        assert_eq(result_df["id"].sort_values(), expected_df, check_index=False)

    @pytest.mark.parametrize("pooling_strategy", ["last_token", "mean"])
    def test_embedding_creator_pooling_strategies(self, tmpdir, pooling_strategy):
        test_text = "The quick brown fox jumps over the lazy dog"
        test_texts = [test_text] * 32
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
        elif pooling_strategy == "mean":
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError("pooling_strategy must be either 'last_token' or 'mean'")

        normed_emb = F.normalize(embeddings, dim=1).cpu()
        normed_emb = normed_emb.squeeze(0)
        embs.append(normed_emb)

    return np.array(embs)
