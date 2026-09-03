import asyncio
from app.config import settings
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.schemas.llm import ChatRequest, ChatMessage


async def main():
    provider = OpenAICompatibleProvider(settings.llm_base_url, settings.groq_api_key)
    resp = await provider.complete(
        ChatRequest(
            messages=[ChatMessage(role="user", content="Say hello in exactly three words.")],
            model=settings.llm_default_model,
        )
    )
    print(resp)


asyncio.run(main())