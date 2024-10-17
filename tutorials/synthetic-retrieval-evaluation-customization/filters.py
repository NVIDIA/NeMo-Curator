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

import itertools
import json
from typing import List, Union

import dask
import dask.dataframe as dd
import fasttext
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from openai import OpenAI

from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import NoWorkerError, load_object_on_worker


# ----------------------------------------------------------------------------80
# ------------------------ EASINESS FILTER -------------------------------------
# ----------------------------------------------------------------------------80
class EasinessFilter(DocumentFilter):

    def __init__(self, cfg: DictConfig, text_fields: List[str] = ["text", "question"]):

        self._name = "easiness_filter"
        if "easiness_filter" in cfg:
            self.filter_cfg = cfg["easiness_filter"]["filter_cfg"]
        else:
            raise Exception("Error: Config doesn't have easiness filter")
        self.base_url = self.filter_cfg.base_url
        self.api_key = self.filter_cfg.api_key
        self.nim_model = self.filter_cfg.nim_model
        self.percentile = self.filter_cfg.percentile
        if self.filter_cfg.truncate:
            self.truncate = self.filter_cfg.truncate
        else:
            self.truncate = "NONE"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.batch_size = self.filter_cfg.batch_size
        self.text_fields = text_fields

    @batched
    def score_document(self, df: Union[pd.DataFrame, dd.DataFrame]):

        document_score = self._calc_similarity_nim(
            df[self.text_fields[0]].to_list(), df[self.text_fields[1]].to_list()
        )
        return pd.Series(document_score)

    @batched
    def keep_document(self, scores: pd.Series):
        filter_threshold = np.percentile(scores, self.percentile)

        return scores <= filter_threshold

    def _get_nim_embedding(self, text, input_type):
        # Obtain embeddings from nim model
        if isinstance(text, list):
            input_ = text
        elif isinstance(text, str):
            input_ = [text]

        try:
            response = self.client.embeddings.create(
                input=input_,
                model=self.nim_model,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": self.truncate},
            )
        except Exception as e:
            print(f"Error: {e}")
            response = None

        if response:
            if isinstance(text, list):
                embeddings = [r.embedding for r in response.data]
            elif isinstance(text, str):
                embeddings = response.data[0].embedding
            return embeddings
        else:
            return []

    def _calc_similarity_nim(self, context, question):
        # cosine similarity
        doc_embed = self._get_nim_embedding(text=context, input_type="passage")
        q_embed = self._get_nim_embedding(text=question, input_type="query")
        if isinstance(context, list) and isinstance(question, list):
            if doc_embed and q_embed:
                sim = np.diag(np.dot(np.array(doc_embed), np.array(q_embed).T))
            else:
                sim = np.zeros(len(context))
        else:
            if doc_embed and q_embed:
                sim = np.dot(doc_embed, q_embed)
            else:
                sim = 0.0

        return sim


# ----------------------------------------------------------------------------80
# ----------------------- Answerability Filter ---------------------------------
# ----------------------------------------------------------------------------80


class AnswerabilityFilter(DocumentFilter):

    def __init__(self, cfg: DictConfig, text_fields: List[str] = ["text", "question"]):

        self._name = "answerability_filter"
        if "answerability_filter" in cfg:
            self.filter_cfg = cfg["answerability_filter"]["filter_cfg"]
        else:
            raise Exception("Error: Config doesn't have answerability filter")
        self.base_url = self.filter_cfg.base_url
        self.api_key = self.filter_cfg.api_key
        self.model_name = self.filter_cfg.model_name
        self.system_prompt = self.filter_cfg.system_prompt
        self.user_prompt_template = self.filter_cfg.user_prompt_template
        self.num_criteria = self.filter_cfg.num_criteria

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.text_fields = text_fields

    @batched
    def score_document(self, df: Union[pd.DataFrame, dd.DataFrame]):
        return df.apply(
            lambda row: self._llm_as_judge(
                row[self.text_fields[0]], row[self.text_fields[1]]
            ),
            axis=1,
        )

    @batched
    def keep_document(self, scores: pd.Series):

        def _keep_document(score: str):
            is_keep = True  # default is to keep
            try:
                json_ans = json.loads(scores)
                for i in range(self.num_criteria):
                    if json_ans[f"criterion_{i+1}"] != "Y":
                        # filter out data if any of the criteria fails
                        is_keep = False  # filter out
                        break
            except Exception as e:
                print(f"Parse error {e}")
                # if there is a parse error, keep the document

            return is_keep

        return scores.apply(_keep_document)

    def _llm_as_judge(self, context: str, question: str):

        user_query = self.system_prompt + "\n\n"
        user_query += self.user_prompt_template.format(
            context=context, question=question
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_query}],
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=True,
            )

            generation = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    generation += chunk.choices[0].delta.content
        except Exception as e:
            print(f"API call error {e}")
            return None  # generation

        return generation


# ----------------------------------------------------------------------------80
