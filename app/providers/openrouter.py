"""OpenRouter generator — thin subclass of the OpenAI-compatible driver."""

from __future__ import annotations

from app.providers.openai_compat import OpenAICompatibleGenerator


class OpenRouterGenerator(OpenAICompatibleGenerator):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(
            model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key, api_key_env="OPENROUTER_API_KEY",
            provider="openrouter",
        )
