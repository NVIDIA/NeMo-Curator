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
    DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
    DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
    DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
    DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
    MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
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

    def _prompt(
        self, model: str, prompt_template: str, prompt_kwargs: dict, model_kwargs: dict
    ) -> List[str]:
        prompt = prompt_template.format(**prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]

        return self.client.query_model(messages=messages, model=model, **model_kwargs)

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
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {llm_response}
                parameter that will be populated with the llm_response value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A parsed list of elements from the original LLM response
        """
        prompt_kwargs["llm_response"] = llm_response
        yaml_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        try:
            parsed_response = yaml.safe_load(yaml_response[0])
        except yaml.error.YAMLError as _:
            raise YamlConversionError(
                f"Error parsing yaml response: {yaml_response[0]}"
            )

        if not isinstance(parsed_response, list):
            raise YamlConversionError(
                f"Error: Parsed response was not a list: {parsed_response}"
            )

        for elem in parsed_response:
            if not isinstance(elem, str):
                raise YamlConversionError(
                    f"Error: Parsed response contains non-string elements in list: {parsed_response}"
                )
            if elem not in llm_response:
                raise YamlConversionError(
                    f"Conversion introduced hallucinations. Original response:\n{llm_response}\nConverted response:\n{parsed_response}\nHallucination:\n{elem}"
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
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {ntopics}
                parameter that will be populated with the ntopics value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def generate_open_qa_from_topic(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of open Q&A questions based on a topic
        Args:
            topic: The topic to generate questions for.
            n_openlines: The number of questions to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def revise_open_qa(
        self,
        openline: str,
        n_revisions: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to revise an open Q&A question a given number of times
        Args:
            openline: An openline to revise
            n_revisions: The number of revisions to generate for the question.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - openline: Will be populated with the openline passed in this function
                - n_revisions: Will be populated with the n_revisions passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["openline"] = openline
        prompt_kwargs["n_revisions"] = n_revisions
        revisions = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    def generate_writing_tasks(
        self,
        topic: str,
        text_material_type: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of writing tasks based on a topic and document type
        Args:
            topic: The topic to generate writing tasks for.
            text_material_type: The type of the document the question should ask to generate (e.g., "Email", "Poem")
            n_openlines: The number of tasks to generate per topic and text material pair.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - topic: Will be populated with the topic passed in this function
                - text_material_type: Will be populated with the text_material_type passed in this function
                - n_openlines: Will be populated with the n_openlines passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["text_material_type"] = text_material_type
        prompt_kwargs["n_openlines"] = n_openlines
        writing_tasks = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return writing_tasks

    def revise_writing_tasks(
        self,
        openline: str,
        n_revisions: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to revise a writing task a given number of times
        Args:
            openline: An openline to revise
            n_revisions: The number of revisions to generate for the task.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - openline: Will be populated with the openline passed in this function
                - n_revisions: Will be populated with the n_revisions passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["openline"] = openline
        prompt_kwargs["n_revisions"] = n_revisions
        revisions = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return revisions

    def generate_closed_qa_instructions(
        self,
        document: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of closed Q&A questions based on a reference document
        Args:
            document: The document to use when generating questions
            n_openlines: The number of questions to generate per document.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - document: Will be populated with the document passed in this function
                - n_openlines: Will be populated with the n_openlines passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["document"] = document
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_math_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        school_level: str,
        model: str,
        prompt_template: str = DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about math
        Args:
            n_macro_topics: The number of macro topics to generate. Can be an integer like 5 or a string like "five".
            school_level: The school level the math questions should be targeted at.
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
                - school_level: Will be populated with the school_level passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        prompt_kwargs["school_level"] = school_level
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_math_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a math macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def classify_math_entity(
        self,
        entity: str,
        model: str,
        prompt_template: str = DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs={},
    ) -> List[str]:
        """
        Prompts an LLM to classify if an entity is related to math
        Args:
            entity: The entity to classify
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - entity: Will be populated with the entity passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["entity"] = entity
        classification_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    def generate_math_problem(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        prompt_template: str = MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of math problems based on a topic
        Args:
            topic: The topic to generate problems for.
            n_openlines: The number of problems to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
                Some example templates found in nemo_curator.synthetic include:
                - MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE
                - MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_python_macro_topics(
        self,
        n_macro_topics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of macro topics about the Python programming language
        Args:
            n_macro_topics: The number of macro topics to generate. Can be an integer like 5 or a string like "five".
            model: The name of the model that should be used to generate the macro topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_macro_topics: Will be populated with the n_macro_topics passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_macro_topics"] = n_macro_topics
        macro_topics = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return macro_topics

    def generate_python_subtopics(
        self,
        macro_topic: str,
        n_subtopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of subtopics relating to a Python macro topic
        Args:
            macro_topic: The macro topic to generate subtopics for.
            n_subtopics: The number of subtopics to generate per macro topic
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_subtopics: Will be populated with the n_subtopics passed in this function
                - macro_topic: Will be populated with the macro_topic passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["n_subtopics"] = n_subtopics
        prompt_kwargs["macro_topic"] = macro_topic
        subtopics_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return subtopics_response

    def classify_python_entity(
        self,
        entity: str,
        model: str,
        prompt_template: str = DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to classify if an entity is related to Python
        Args:
            entity: The entity to classify
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - entity: Will be populated with the entity passed in this function
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["entity"] = entity
        classification_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return classification_response

    def generate_python_problem(
        self,
        topic: str,
        n_openlines: Union[str, int],
        model: str,
        language="Python",
        prompt_template: str = PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of coding problems based on a topic
        Args:
            topic: The topic to generate problems for.
            n_openlines: The number of problems to generate per topic.
            model: The name of the model that should be used to generate the response.
                Must be available in the LLMClient passed in the constructor.
            language: The programming language to target when generating these questions.
            prompt_template: A format string of the prompt to use. It must have the following parameters:
                - n_openlines: Will be populated with the n_subtopics passed in this function
                - topic: Will be populated with the topic passed in this function
                - language: Will be populated with the language passed in this function
                Some example templates found in nemo_curator.synthetic include:
                - PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE
                - PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt_kwargs["topic"] = topic
        prompt_kwargs["n_openlines"] = n_openlines
        prompt_kwargs["language"] = language
        openline_response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return openline_response

    def generate_data_assistance_openlines(self):
        pass

    def generate_dialogue(
        self,
        openline: str,
        user_model: str,
        assistant_model: str,
        n_user_turns: int = 3,
        prompt_template: str = DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        user_model_kwargs: dict = {},
        assistant_model_kwargs: dict = {},
    ) -> List[dict]:
        """
        Prompts an LLM to generate a dialogue based on a given openline.
        The LLM will alternate impersonating the user and the assistant.
        Args:
            openline: The openline that will comprise the first user turn.
            user_model: The model that will be impersonating the user.
                Must be available in the LLMClient passed in the constructor.
            assistant_model: The model that will be impersonating the assistant
                Must be available in the LLMClient passed in the constructor.
            n_user_turns: The number of user turns to go through. The openline counts as 1 user turn.
                Therefore, if there are 3 user turns, 2 will be generated by the LLM impersonating the user.
            prompt_template: A format string of the prompt to use when impersonating the user.
                It must have the following parameters:
                - converstation_history: Will be populated with a formatted history of the dialogue up to that point.
                Some example templates found in nemo_curator.synthetic include:
                - DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            user_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the user.
            assistant_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the assistant.
        Returns:
            A conversation between a User and Assistant
        """
        conversation_history = [{"role": "user", "content": openline}]
        first_assistant_response = self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )
        for _ in range(n_user_turns - 1):
            user_response = self._impersonate_user(
                conversation_history=conversation_history,
                model=user_model,
                prompt_template=prompt_template,
                prompt_kwargs=prompt_kwargs,
                model_kwargs=user_model_kwargs,
            )
            conversation_history.append({"role": "user", "content": user_response})
            assistant_response = self.client.query_model(
                messages=conversation_history,
                model=assistant_model,
                **assistant_model_kwargs,
            )[0]
            conversation_history.append(
                {"role": "assistant", "content": assistant_response}
            )

        return conversation_history

    def generate_two_turn_prompt(
        self,
        openline: str,
        user_model: str,
        assistant_model: str,
        prompt_template: str = DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        user_model_kwargs: dict = {},
        assistant_model_kwargs: dict = {},
    ) -> List[dict]:
        """
        Prompts an LLM to generate a response as an assistant, then as the user based on a given openline.
        The conversation will look like "User -> Assistant -> User"
        Args:
            openline: The openline that will comprise the first user turn.
            user_model: The model that will be impersonating the user.
                Must be available in the LLMClient passed in the constructor.
            assistant_model: The model that will be impersonating the assistant
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use when impersonating the user.
                It must have the following parameters:
                - converstation_history: Will be populated with a formatted history of the dialogue up to that point.
                Some example templates found in nemo_curator.synthetic include:
                - DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE
                - DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
                None are needed for the default template.
            user_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the user.
            assistant_model_kwargs: Any additional keyword arguments that should be passed to the
                LLMClient.query_model call for the assistant.
        Returns:
            A conversation between a User and Assistant
        """
        conversation_history = [{"role": "user", "content": openline}]
        first_assistant_response = self.client.query_model(
            messages=conversation_history,
            model=assistant_model,
            **assistant_model_kwargs,
        )[0]
        conversation_history.append(
            {"role": "assistant", "content": first_assistant_response}
        )

        user_response = self._impersonate_user(
            conversation_history=conversation_history,
            model=user_model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=user_model_kwargs,
        )
        conversation_history.append({"role": "user", "content": user_response})

        return conversation_history

    def _impersonate_user(
        self,
        conversation_history: List[dict],
        model: str,
        prompt_template: str,
        prompt_kwargs: dict,
        model_kwargs: dict,
    ) -> str:
        # Convert the conversation history to a string
        history_str = ""
        for turn in conversation_history:
            history_str += f"{turn['role'].capitalize()}: {turn['content']}"
        prompt_kwargs["conversation_history"] = history_str
        response = self._prompt(
            model=model,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        return response[0]
