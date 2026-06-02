"""OpenAICompatibleGenerator — one driver for every OpenAI-compatible endpoint.

DeepSeek, OpenRouter, OpenAI, Gemini's OpenAI-compat layer, and local servers
(vLLM / Ollama / LM Studio / TGI) all speak the same wire protocol, so a single
class — parameterized by base_url + api key — covers ~all of them. Vendor-specific
generators are thin subclasses that just fill in base_url and the key env var.
"""

from __future__ import annotations

import os
from typing import Sequence

from openai import AsyncOpenAI

from app.models import Evidence
from app.providers.offline import _content  # reuse content-token overlap for grounding

_SYSTEM = (
    "You are a closed-book examiner. Answer the user's question using ONLY the "
    "evidence provided. Cite every claim with the source in square brackets like "
    "[doc_title]. If the evidence does not support an answer, reply exactly: "
    "'I don't have sufficient evidence to answer this confidently.' "
    "Do not use outside knowledge. Be concise — at most 3 sentences."
)


class OpenAICompatibleGenerator:
    def __init__(
        self, model: str, *, base_url: str,
        api_key: str | None = None, api_key_env: str | None = None,
        provider: str = "openai-compatible",
    ) -> None:
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        if not key:
            raise RuntimeError(
                f"No API key for {provider}. Set {api_key_env or 'the API key env var'}."
            )
        self.model = model
        self.name = f"{provider}:{model}"
        self.client = AsyncOpenAI(base_url=base_url, api_key=key)

    @staticmethod
    def _context(evidence: Sequence[Evidence]) -> str:
        return "\n\n".join(
            f"[{e.doc_title} §{'/'.join(e.section_path) or '-'}]\n{e.text}" for e in evidence
        )

    async def generate(self, query: str, evidence: Sequence[Evidence]) -> str:
        if not evidence:
            return "I don't have sufficient evidence to answer this confidently."
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"Evidence:\n{self._context(evidence)}\n\nQuestion: {query}"},
        ]
        # Reasoning models spend tokens on hidden reasoning; give headroom and retry
        # once if content returns empty (a common reasoning-model / free-tier hiccup).
        for max_tokens in (2000, 3000):
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        return "I don't have sufficient evidence to answer this confidently."

    async def complete(self, prompt: str, *, max_tokens: int = 1500) -> str:
        """Generic single-prompt completion — used by ingestion refine + representations.

        Present only on real-LLM generators; the offline ExtractiveGenerator omits it
        on purpose, so callers fall back to deterministic behavior via getattr().
        """
        resp = await self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],  # type: ignore[arg-type]
        )
        return (resp.choices[0].message.content or "").strip()

    async def judge_grounding(self, query: str, evidence: Sequence[Evidence]) -> tuple[float, bool]:
        if not evidence:
            return (0.0, False)
        q = _content(query)
        covered: set[str] = set()
        for e in evidence:
            covered |= (_content(e.text) & q)
        coverage = len(covered) / max(len(q), 1)
        return (round(min(1.0, 0.4 + coverage), 3), False)
