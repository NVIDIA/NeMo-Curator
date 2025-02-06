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

from typing import List, Optional

from transformers import AutoTokenizer

from nemo_curator.services import LLMClient
from nemo_curator.synthetic.prompts import (
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


class NemotronCC:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-CC paper (https://arxiv.org/abs/2412.02595).
    """

    def __init__(self, llm_client: LLMClient, tokenizer: AutoTokenizer) -> None:
        self.client = llm_client
        self.tokenizer = tokenizer

    def _prompt(
        self,
        model: str,
        prompt_template: str,
        system_prompt: str,
        prompt_kwargs: dict,
        model_kwargs: dict,
    ) -> List[str]:
        prompt = prompt_template.format(**prompt_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self.client.query_model(messages=messages, model=model, **model_kwargs)

    def get_wikipedia_prefix_token_count(self) -> int:
        user_prompt = WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE.format(
            **{"document": "placeholder"}
        )
        messages = [
            {"role": "system", "content": NEMOTRON_CC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prefix = self.tokenizer.apply_chat_template(messages)

        return len(self.tokenizer.encode(prefix))

    def rewrite_to_wikipedia_style(
        self,
        document: str,
        model: str,
        prompt_template: str = WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        system_prompt: str = NEMOTRON_CC_SYSTEM_PROMPT,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> str:
        prompt_kwargs["document"] = document
        return self._prompt(
            model, prompt_template, system_prompt, prompt_kwargs, model_kwargs
        )
