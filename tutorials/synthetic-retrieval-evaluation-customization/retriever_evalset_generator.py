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

import asyncio
import hashlib
import importlib
import os
import re
import secrets
from abc import ABC, abstractmethod
from typing import Any

import dask.array as da
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from distributed import Client
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from tqdm.dask import TqdmCallback

from nemo_curator import AsyncOpenAIClient, OpenAIClient
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.synthetic import AsyncNemotronGenerator, NemotronGenerator
from nemo_curator.synthetic.generator import SyntheticDataGenerator
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import get_client, get_current_client
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.module_utils import is_batched
from nemo_curator.utils.script_utils import ArgumentHelper


class RetrieverEvalSetGenerator(SyntheticDataGenerator):

    def __init__(
        self,
        pipeline_config: DictConfig = None,
    ):
        super().__init__()
        self._name = self.__class__.__name__
        self.cfg = pipeline_config

        # synchronous
        self.openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.cfg["qa_generator"]["api_key"],
        )
        self.client = OpenAIClient(self.openai_client)
        self.generator = NemotronGenerator(self.client)

        # TODO asynchronous
        self.async_openai_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.cfg["qa_generator"]["api_key"],
        )
        self.async_client = AsyncOpenAIClient(self.async_openai_client)
        self.async_generator = AsyncNemotronGenerator(self.async_client)

        self._init_pipeline_params()

    def load_pipeline_config(self, cfg_path: str):
        self.cfg = OmegaConf.load(cfg_path)

    def _validate_config(self):
        return True  # TODO complete this

    def _init_pipeline_params(self):

        if self._validate_config():
            self.sys_prompt = self.cfg["qa_generator"]["generate_config"][
                "system_prompt"
            ]
            self.user_prompt_template = self.cfg["qa_generator"]["generate_config"][
                "user_prompt_template"
            ]
            self.generator_model = self.cfg["qa_generator"]["model"]
            self.generator_model_kwargs = self.cfg["qa_generator"]["model_config"]
            self.num_qs = self.cfg["qa_generator"]["generate_config"]["num_questions"]
            self.easiness_filter_cfg = self.cfg["easiness_filter"]["filter_cfg"]
        else:
            raise Exception("Validation Error: incorrect pipeline config file")

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:

        df = dataset.df

        df["llm_response"] = df["text"].apply(
            self.generate, meta=("llm_response", "str")
        )
        df["qa_pairs"] = df["llm_response"].apply(
            self.parse_response, meta=("qa_pairs", "object")
        )

        df = df.explode("qa_pairs").reset_index(drop=True)
        df["question"] = df["qa_pairs"].apply(
            lambda x: x["question"], meta=("question", "str")
        )
        df["question-id"] = df["question"].apply(
            self._get_random_hash, meta=("question-id", "str")
        )
        df["answer"] = df["qa_pairs"].apply(
            lambda x: x["answer"], meta=("answer", "str")
        )
        df["score"] = df["question"].apply(lambda x: 1, meta=("score", "int"))

        return DocumentDataset(
            df[["_id", "text", "title", "question-id", "question", "answer", "score"]]
        )

        # qa_pairs = []
        # for _, row in tqdm(df.iterrows()):
        #     qa_pairs.append(self.parse_response(self.generate(row['text'])))
        # for qa in qa_pairs:
        #     result.append({"_id": row["_id"],
        #                    "text" : row["text"],
        #                    "title": row["title"],
        #                    "question-id": self._get_random_hash(qa['question']),
        #                    "question": qa["question"],
        #                    "answer": qa["answer"],
        #                    "score": 1}
        #     )

        # return DocumentDataset(dd.DataFrame(result))

        # df['qa_pairs'] =  da.from_array(qa_pairs, chunks=df.npartitions)
        # asynchronous:
        # return asyncio.run(self._run_from_source(dataset))

    # def parse_response(self, llm_response: str) -> Any:

    #     return asyncio.gather(self._parse_response(llm_response))

    def parse_response(self, llm_response: str) -> Any:
        qa_pairs = []
        qa_list = llm_response.split("Question")[1:]
        try:
            for qa in qa_list:
                qas = qa.split("Answer")
                q = qas[0].split(":")[1].strip()
                if re.search("Explanation", qas[1]):
                    a = qas[1].split("Explanation")[0].split(":")[1].strip()
                    explanation = qas[1].split("Explanation")[1].strip()  # Not used
                else:
                    a = qas[1].split(":")[1].strip()
                qa_pairs.append({"question": q, "answer": a})
        except Exception as e:
            print(f"error: {e}")
        return qa_pairs

    def generate(self, doc_text):

        response = self.generator.generate_closed_qa_instructions(
            document=doc_text,
            prompt_template=self.sys_prompt + "\n" + self.user_prompt_template,
            n_openlines=self.num_qs,
            model=self.generator_model,
            model_kwargs=self.generator_model_kwargs,
        )

        return response[0]

    def _get_random_hash(self, question: str):
        """Generate random hash for synthetic question IDs"""
        # Generate a random string
        random_string = secrets.token_hex(
            16
        )  # Generates a secure, random string of 16 bytes hex-encoded

        # Hash the random string using SHA-256
        hash_object = hashlib.sha256(
            random_string.encode()
        )  # Encode the string to bytes
        hex_dig = hash_object.hexdigest()
        return hex_dig

    # async def _run_from_source(self, dataset: DocumentDataset) -> DocumentDataset:

    #     df = dataset.df
    #     qa_pairs_list = []
    #     for _, row in df.iterrows():
    #         llm_response = await self.generate(row['text'])
    #         qa_pairs_list.append(await self.parse_response(llm_response))

    #     for i in tqdm(range(len(qa_pairs_list))):
    #         result = await tqdm.gather(qa_pairs_list[i])
    #         print (result)

    #     return dataset
