from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union


class LLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests synchronously
    """

    @abstractmethod
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
        raise NotImplementedError("Subclass of LLMClient must implement 'query_model'")


class AsyncLLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests asynchronously
    """

    @abstractmethod
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
        raise NotImplementedError("Subclass of LLMClient must implement 'query_model'")
