import httpx

from app.llm.base import LLMProvider
from app.schemas.llm import ChatRequest, ChatResponse, TokenUsage


class OpenAICompatibleProvider(LLMProvider):
    """Adapter for any OpenAI-compatible /chat/completions API (Groq, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        provider_name: str = "openai-compatible",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._name = provider_name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        body: dict = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage")
        usage = (
            TokenUsage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"],
            )
            if usage_data
            else None
        )
        return ChatResponse(content=choice, model=data["model"], usage=usage)
