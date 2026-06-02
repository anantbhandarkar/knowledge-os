"""DeepSeek generator — direct DeepSeek API (OpenAI-compatible).

Set DEEPSEEK_API_KEY and run with KOS_MODE=api KOS_PROVIDER=deepseek.
Models: "deepseek-chat" (V3, fast) or "deepseek-reasoner" (R1, deeper reasoning).

NOTE: DeepSeek serves chat/reasoning only — it has no embeddings endpoint, so the
embedder stays a separate provider (HashEmbedder by default, or a real one in api mode).
"""

from __future__ import annotations

from app.providers.openai_compat import OpenAICompatibleGenerator


class DeepSeekGenerator(OpenAICompatibleGenerator):
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None) -> None:
        super().__init__(
            model,
            base_url="https://api.deepseek.com",
            api_key=api_key, api_key_env="DEEPSEEK_API_KEY",
            provider="deepseek",
        )
