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

import json
import warnings
from typing import Dict, List, Optional

from openai import OpenAI

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.utils.distributed_utils import load_object_on_worker
from nemo_curator.utils.llm_pii_utils import (
    JSON_SCHEMA,
    PII_LABELS,
    SYSTEM_PROMPT,
    redact,
    validate_entity,
)

__all__ = ["LLMPiiModifier"]


class LLMInference:
    """A class for redacting PII via LLM inference"""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "meta/llama-3.1-70b-instruct",
        system_prompt: str = SYSTEM_PROMPT,
        pii_labels: List[str] = PII_LABELS,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt[model]
        self.pii_labels = pii_labels

    def infer(self, text: str) -> List[Dict[str, str]]:
        """Invoke LLM to get PII entities"""

        text = text.strip()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # The field guided_json is unsupported at the root level
            # and must be included in the nvext object field
            extra_body={"nvext": {"guided_json": JSON_SCHEMA}},
            stream=False,
            max_tokens=4096,
        )

        assistant_message = response.choices[0].message.content

        # Parse results
        try:
            entities = json.loads(assistant_message)
            if not entities:
                # LLM returned valid JSON but no entities discovered
                return []
            else:
                # Check that each entity returned is valid
                return [e for e in entities if validate_entity(e, text)]
        except json.decoder.JSONDecodeError:
            return []


class LLMPiiModifier(DocumentModifier):
    """
    This class is the entry point to using the LLM-based PII de-identification module.
    It works with the `Modify` functionality as shown below:

    dataframe = pd.DataFrame({"text": ["Sarah and Ryan went out to play", "Jensen is the CEO of NVIDIA"]})
    dd = dask.dataframe.from_pandas(dataframe, npartitions=1)
    dataset = DocumentDataset(dd)

    modifier = LLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
        # The user may engineer a custom prompt if desired
        system_prompt=SYSTEM_PROMPT,
        pii_labels=PII_LABELS,
        language="en",
    )

    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    modified_dataset.df.to_json("output_files/*.jsonl", lines=True, orient="records")

    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "meta/llama-3.1-70b-instruct",
        system_prompt: str = SYSTEM_PROMPT,
        pii_labels: List[str] = PII_LABELS,
        language: str = "en",
    ):
        """
        Initialize the LLMPiiModifier

        Args:
            base_url (str): The base URL for the user's NIM
            api_key (Optional[str]): The API key for the user's NIM, if needed.
                Default is None.
            model (str): The model to use for the LLM.
                Default is "meta/llama-3.1-70b-instruct".
            system_prompt (str): The system prompt to feed into the LLM.
                Default prompt has been fine-tuned for "meta/llama-3.1-70b-instruct".
            pii_labels (List[str]): The PII labels to identify and remove from the text.
                See documentation for full list of PII labels.
            language (str): The language to use for the LLM.
                Default is "en" for English. If non-English, it is recommended
                to provide a custom system prompt.

        """
        super().__init__()

        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.pii_labels = pii_labels
        self.language = language

        if self.language != "en" and self.system_prompt is SYSTEM_PROMPT:
            warnings.warn(
                "The default system prompt is only available for English. "
                "For other languages, please provide a custom system prompt."
            )
        if self.model not in SYSTEM_PROMPT:
            warnings.warn(
                f"No system prompt has been defined for model {model}. "
                "Default system prompt will be used."
            )
            self.system_prompt[self.model] = SYSTEM_PROMPT[
                "meta/llama-3.1-70b-instruct"
            ]

    def modify_document(self, text: str):
        inferer = load_object_on_worker("inferer", self.load_inferer, {})
        pii_entities = inferer.infer(text)
        text_redacted = redact(text, pii_entities)
        return text_redacted

    def load_inferer(self):
        """Helper function to load the LLM"""
        inferer: LLMInference = LLMInference(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            pii_labels=self.pii_labels,
        )

        return inferer
