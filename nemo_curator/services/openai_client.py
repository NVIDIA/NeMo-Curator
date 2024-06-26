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
        stop: Union[Optional[str], List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            top_p=top_p,
        )


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
        stop: Union[Optional[str], List[str]] = None,
        top_p: Optional[float] = None
    ) -> str:
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            top_p=top_p,
        )
