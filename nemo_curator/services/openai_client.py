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
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN
    ) -> List[str]:
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]


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
        stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN
    ) -> List[str]:
        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        return [choice.message.content for choice in response.choices]
