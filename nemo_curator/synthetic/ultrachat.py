from typing import List, Optional, Union

from nemo_curator.services.model_client import AsyncLLMClient, LLMClient
from nemo_curator.synthetic.prompts import DEFAULT_META_TOPICS_PROMPT_TEMPLATE


class UltraChatGenerator:
    """
    Provides a collection of methods for generating synthetic data inspired
    by the UltraChat paper: https://arxiv.org/abs/2305.14233
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.client = llm_client

    def generate_world_question_openlines(self):
        pass

    def generate_meta_topics(
        self,
        ntopics: Union[int, str],
        model: str,
        prompt_template: str = DEFAULT_META_TOPICS_PROMPT_TEMPLATE,
        prompt_kwargs: dict = {},
        model_kwargs: dict = {},
    ) -> List[str]:
        """
        Prompts an LLM to generate a list of meta topics about the world
        Args:
            ntopics: The number of meta topics to generate. Can be an integer like 5 or a string like "five".
                It is used where it is referenced in prompt_template
            model: The name model that should be used to generate the meta topics.
                Must be available in the LLMClient passed in the constructor.
            prompt_template: A format string of the prompt to use. It must have a {ntopics}
                parameter that will be populated with the ntopics value passed in this function.
            prompt_kwargs: Any additional keyword arguments that should be passed to the prompt template.
            model_kwargs: Any additional keyword arguments that should be passed to the LLMClient.query_model call.
        Returns:
            A list of responses from the LLM. The list is only greater than length 1 if n > 1 is set in model_kwargs.
        """
        prompt = prompt_template.format(ntopics=ntopics, **prompt_kwargs)
        messages = [{"role": "user", "content": prompt}]
        meta_topics = self.client.query_model(
            messages=messages, model=model, **model_kwargs
        )

        return meta_topics

    def generate_subtopics(self, meta_topics: List[str], model: str):
        pass

    def generate_questions(self, subtopics):
        pass

    def generate_creative_openlines(self):
        pass

    def generate_data_assistance_openlines(self):
        pass
