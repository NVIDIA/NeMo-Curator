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

import importlib
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from tqdm import tqdm

from nemo_curator import OpenAIClient
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.synthetic import NemotronGenerator
from nemo_curator.synthetic.generator import SyntheticDataGenerator
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.module_utils import is_batched


class RetrieverEvalSetGenerator(SyntheticDataGenerator):

    def __init__(self, pipeline_config: DictConfig = None):
        super().__init__()
        self._name = self.__class__.__name__
        self.cfg = pipeline_config

        # synchronous
        self.openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],  # TODO api_key
        )
        self.client = OpenAIClient(self.openai_client)
        self.generator = NemotronGenerator(self.client)

        # TODO asynchronous

    def load_pipeline_config(self, cfg_path: str):
        self.cfg = OmegaConf.load(cfg_path)

    def validate_config(self):
        return True  # TODO complete this

    def init_pipeline_params(self):

        if self.validate_config:
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

    def run(self, dataset: DocumentDataset) -> DocumentDataset:

        df = dataset.df
        df["llm_response"] = df["text"].apply(
            self._generate, meta=("llm_response", "str")
        )
        df["qa_pairs"] = df["llm_response"].apply(
            self._parse_response, meta=("qa_pairs", "object")
        )
        df = df.explode("qa_pairs").reset_index(drop=True)
        df["question"] = df["qa_pairs"].apply(
            lambda x: x["question"], meta=("question", "str")
        )
        df["answer"] = df["qa_pairs"].apply(
            lambda x: x["answer"], meta=("answer", "str")
        )
        return DocumentDataset(df[["_id", "text", "title", "question", "answer"]])

    def _parse_response(self, llm_response: str) -> Any:
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

    def _generate(self, doc_text):
        return self.generator.generate_closed_qa_instructions(
            document=doc_text,
            prompt_template=self.sys_prompt + "\n" + self.user_prompt_template,
            n_openlines=self.num_qs,
            model=self.generator_model,
            model_kwargs=self.generator_model_kwargs,
        )[0]
