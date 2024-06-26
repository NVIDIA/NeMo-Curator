from typing import Iterable, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

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
        max_tokens: Optional[int] = None,
        n: Optional[int] = 1,
        stop: Union[Optional[str], List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
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
        max_tokens: Optional[int] = None,
        n: Optional[int] = 1,
        stop: Union[Optional[str], List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
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
