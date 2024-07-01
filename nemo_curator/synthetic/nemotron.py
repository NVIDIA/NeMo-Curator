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
from typing import List, Optional, Union

import yaml

from nemo_curator.services.model_client import AsyncLLMClient, LLMClient
from nemo_curator.synthetic.error import YamlConversionError
from nemo_curator.synthetic.prompts import (
    DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
)


class NemotronGenerator:
    """
    Provides a collection of methods for generating synthetic data
    described in the Nemotron-4 340B Technical Report
    (https://arxiv.org/abs/2406.11704v1) and inspired by the
    UltraChat paper (https://arxiv.org/abs/2305.14233)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.client = llm_client

    def convert_response_to_yaml_list(
        self,
        llm_response: str,
        model: str,
        prompt_template: str = DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Converts a response of an LLM to a list of strings by querying an LLM
        Args:
            llm_response: The original unformatted response of the LLM
            model: The name model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {llm_response}
                parameter that will be populated with the llm_response value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A parsed list of elements from the original LLM response
        """
        prompt = prompt_template.format(llm_response=llm_response, **prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]
        yaml_response = self.client.query_model(
            messages=messages, model=model, **model_kwargs
        )
        try:
            parsed_response = yaml.safe_load(yaml_response[0])
        except yaml.scanner.ScannerError as _:
            raise YamlConversionError(
                f"Error parsing yaml response: {yaml_response[0]}"
            )

        # Ensure there are no additional hallucinations introduced
        hallucination_free = all(elem in llm_response for elem in parsed_response)
        if not hallucination_free:
            raise YamlConversionError(
                f"Conversion introduced hallucinations. Original response:\n{llm_response}\nConverted response:\n{parsed_response}"
            )

        return parsed_response

    def generate_world_question_openlines(self):
        pass

    def generate_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about the world
        Args:
            n_macro_topics: The number of macro topics to generate. Can be an integer like 5 or a string like "five".
                It is used where it is referenced in prompt_template
            model: The name model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {ntopics}
                parameter that will be populated with the ntopics value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt = prompt_template.format(n_macro_topics=n_macro_topics, **prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]
        macro_topics = self.client.query_model(
            messages=messages, model=model, **model_kwargs
        )

        return macro_topics

    def generate_subtopics(
        self,
        macro_topics: List[str],
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        """
        Prompts an LLM to generate a list of subtopics relating to a macro topic
        Args:
            macro_topics: A list of macro topics to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with an element of the macro_topics list passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM for each macro topic. The outer list will have the same length
            as macro_topics, while the inner list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        subtopics = []
        for macro_topic in macro_topics:
            prompt = prompt_template.format(
                n_subtopics=n_subtopics, macro_topic=macro_topic, **prompt_kwargs
            )
            messages = [{"role": "user", "content": prompt}]
            subtopics_response = self.client.query_model(
                messages=messages, model=model, **model_kwargs
            )
            subtopics.append(subtopics_response)

        return subtopics

    def generate_open_qa_from_topics(
        self,
        topics: List[str],
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        """
        Prompts an LLM to generate a list of open Q&A questions based on topics
        Args:
            topics: A list of topics to generate questions for.
            n_openlines: The number of questions to generate per topic.
            model: The name model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with an element of the topics list passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM for each topic. The outer list will have the same length
            as topics, while the inner list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        openlines = []
        for topic in topics:
            prompt = prompt_template.format(
                n_openlines=n_openlines, topic=topic, **prompt_kwargs
            )
            messages = [{"role": "user", "content": prompt}]
            subtopics_response = self.client.query_model(
                messages=messages, model=model, **model_kwargs
            )
            openlines.append(subtopics_response)

        return openlines

    def generate_creative_openlines(self):
        pass

    def generate_data_assistance_openlines(self):
        pass
