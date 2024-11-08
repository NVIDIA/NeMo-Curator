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

import dask
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from tutorials.synthetic_retrieval_evaluation_customization.filters import (
    AnswerabilityFilter,
    EasinessFilter,
)
from tutorials.synthetic_retrieval_evaluation_customization.retriever_evalset_generator import (
    RetrieverEvalSetGenerator,
)

from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules import Filter, Score, ScoreFilter, Sequential
from nemo_curator.modules.config import RetrieverEvalSDGConfig


@pytest.fixture
def get_original_data():
    docs = [
        {
            "_id": "930220d64a44c223df83e0caf09013fffdf4c19c1f501f035862984979928b29",
            "text": "The Eiffel Tower is an iconic landmark of Paris, France. It was designed by the engineer Gustave Eiffel and built for the 1889 Exposition Universelle (World's Fair) to celebrate the 100th anniversary of the French Revolution.",
            "title": "Eiffel Tower - A French Icon",
        },
        {
            "_id": "5cdca9fa81b6c4d8a1a1159610c98b2bffae498dad36c90639413bf22e5a4154",
            "text": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states and empires against raids and invasions from various nomadic groups.",
            "title": "The Great Wall of China - Ancient Protection",
        },
    ]
    return DocumentDataset.from_pandas(pd.DataFrame(docs))


@pytest.fixture
def get_generated_data():
    docs = [
        {
            "_id": "930220d64a44c223df83e0caf09013fffdf4c19c1f501f035862984979928b29",
            "text": "The Eiffel Tower is an iconic landmark of Paris, France. It was designed by the engineer Gustave Eiffel and built for the 1889 Exposition Universelle (World's Fair) to celebrate the 100th anniversary of the French Revolution.",
            "title": "Eiffel Tower - A French Icon",
            "question-id": "d9be8cb0693a354b2ba8ddd1e86c9df57db97f33e03cb33c972f8efed4084f8b",
            "question": "What is the significance of the Eiffel Tower in relation to the French Revolution?",
            "answer": "The Eiffel Tower was built to celebrate the 100th anniversary of the French Revolution.",
        },
        {
            "_id": "5cdca9fa81b6c4d8a1a1159610c98b2bffae498dad36c90639413bf22e5a4154",
            "text": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states and empires against raids and invasions from various nomadic groups.",
            "title": "The Great Wall of China - Ancient Protection",
            "question-id": "e1f6e179a883a7f108d566a582a159322b2eb2b8e0d51fd78cafc72373e4be2b",
            "question": "What is the purpose of the Great Wall of China?",
            "answer": "The purpose of the Great Wall of China is to protect the Chinese states and empires against raids and invasions from various nomadic groups.",
        },
        {
            "_id": "35f822b0b38de133b815139affac94d57d0c7d35de6e11d0e52a69d416c1d248",
            "text": "Machu Picchu is a 15th-century Inca citadel situated on a mountain ridge above the Sacred Valley in Peru. It is the most famous icon of Inca civilization, known for its sophisticated dry-stone walls that fuse huge blocks without the use of mortar.",
            "title": "Machu Picchu - Lost City of the Incas",
            "question-id": "1a60e50066c938784db3d49c41e470c197b3fc30afa07957575a1d8a34a34230",
            "question": "What is Machu Picchu renowned for in terms of its architecture?",
            "answer": "Machu Picchu is renowned for its sophisticated dry-stone walls that fuse huge blocks without the use of mortar.",
        },
    ]
    return DocumentDataset.from_pandas(pd.DataFrame(docs))


@pytest.fixture
def get_config():
    cfg = RetrieverEvalSDGConfig.from_yaml(
        "./tutorials/synthetic_retrieval_evaluation_customization/config/config.yaml"
    )
    cfg.api_key = os.environ.get("NVIDIA_API_KEY")
    return cfg


class TestSDGModule:
    def test_generator(self, get_original_data, get_config):

        num_questions = get_config.num_questions
        original_df = get_original_data.df.compute()
        generator = RetrieverEvalSetGenerator(get_config)
        generated_dataset = generator(get_original_data)
        generated_df = generated_dataset.df.compute()
        assert "question" in generated_df
        assert "answer" in generated_df
        assert generated_df.shape[0] == num_questions * original_df.shape[0]
        assert "question-id" in generated_df
        assert "_id" in generated_df
        assert generated_df["_id"].iloc[0] == original_df["_id"].iloc[0]


class TestSDGFilterModule:
    def test_easiness_filter(self, get_generated_data, get_config):

        ef = EasinessFilter(get_config)
        easiness_filter = ScoreFilter(
            ef, text_field=["text", "question"], score_field="easiness_scores"
        )

        org_df = get_generated_data.df.compute()
        filtered_dataset = easiness_filter(get_generated_data)
        filtered_df = filtered_dataset.df.compute()
        assert "easiness_scores" in filtered_df
        assert org_df.shape[0] > filtered_df.shape[0]

    def test_answerability_filter(self, get_generated_data, get_config):

        af = AnswerabilityFilter(get_config)
        answerability_filter = ScoreFilter(
            af, text_field=["text", "question"], score_field="answerability_scores"
        )
        org_df = get_generated_data.df.compute()
        filtered_dataset = answerability_filter(get_generated_data)
        filtered_df = filtered_dataset.df.compute()
        assert "answerability_scores" in filtered_df
        assert org_df.shape[0] >= filtered_df.shape[0]
