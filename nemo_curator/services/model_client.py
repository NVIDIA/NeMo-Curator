from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests synchronously
    """

    @abstractmethod
    def query_model(self, user_input: str) -> str:
        raise NotImplementedError("Subclass of LLMClient must implement 'query_model'")


class AsyncLLMClient(ABC):
    """
    Interface representing a client connecting to an LLM inference server
    and making requests asynchronously
    """

    @abstractmethod
    async def query_model(self, user_input: str) -> str:
        raise NotImplementedError("Subclass of LLMClient must implement 'query_model'")
