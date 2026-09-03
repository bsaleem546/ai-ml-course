from abc import ABC, abstractmethod

from app.schemas.llm import ChatRequest, ChatResponse


class LLMProvider(ABC):
    """Vendor-neutral interface for a chat LLM.

    The rest of the app depends only on this class and the ChatRequest/
    ChatResponse schemas — never on a provider SDK or a vendor-shaped object.
    Each concrete provider (Ollama, a hosted API, a fake for tests) is a subclass.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider, e.g. "ollama". Used by the
        provider registry (task 4) and in logs."""
        ...

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to the model and return a normalized response.

        Implementations translate `request` into the provider's wire format,
        make the call, and map the result back into a ChatResponse (filling
        `usage` when the provider reports token counts, else leaving it None).
        """
        ...
