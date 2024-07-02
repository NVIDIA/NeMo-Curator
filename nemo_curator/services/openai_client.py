from typing import Iterable, List, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai._types import NOT_GIVEN, NotGiven

from .model_client import AsyncLLMClient, LLMClient


class OpenAIClient(LLMClient):
    """
    A wrapper around OpenAI's Python client for querying models
    """

    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    ) -> List[str]:
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]

    def query_reward_model(self, *, messages: Iterable, model: str) -> dict:
        """
        Prompts an LLM Reward model to score a conversation between a user and assistant
        Args:
            messages: The conversation to calculate a score for.
                Should be formatted like:
                    [{"role": "user", "content": "Write a sentence"}, {"role": "assistant", "content": "This is a sentence"}, ...]
            model: The name of the model that should be used to calculate the reward.
                Must be a reward model, cannot be a regular LLM.
        Returns:
            A mapping of score_name -> score
        """
        response = self.client.chat.completions.create(messages=messages, model=model)

        try:
            message = response.choices[0].message[0]
        except TypeError as _:
            raise ValueError(f"{model} is not a reward model.")

        metrics = [metric.split(":") for metric in message.content.split(",")]
        scores = {category: float(score) for category, score in metrics}

        return scores


class AsyncOpenAIClient(AsyncLLMClient):
    """
    A wrapper around OpenAI's Python async client for querying models
    """

    def __init__(self, async_openai_client: AsyncOpenAI) -> None:
        self.client = async_openai_client

    async def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    ) -> List[str]:
        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]

    async def query_reward_model(self, *, messages: Iterable, model: str) -> dict:
        """
        Prompts an LLM Reward model to score a conversation between a user and assistant
        Args:
            messages: The conversation to calculate a score for.
                Should be formatted like:
                    [{"role": "user", "content": "Write a sentence"}, {"role": "assistant", "content": "This is a sentence"}, ...]
            model: The name of the model that should be used to calculate the reward.
                Must be a reward model, cannot be a regular LLM.
        Returns:
            A mapping of score_name -> score
        """
        response = await self.client.chat.completions.create(
            messages=messages, model=model
        )

        try:
            message = response.choices[0].message[0]
        except TypeError as _:
            raise ValueError(f"{model} is not a reward model.")

        metrics = [metric.split(":") for metric in message.content.split(",")]
        scores = {category: float(score) for category, score in metrics}

        return scores
