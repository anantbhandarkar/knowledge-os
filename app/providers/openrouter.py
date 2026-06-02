"""OpenRouterGenerator — real LLM generation via OpenRouter (OpenAI-compatible API).

Used when KOS_MODE=api. Enforces the closed-book examiner prompt: answer ONLY from
the provided evidence, cite [doc_title], and refuse when unsupported. The grounding
judge is a fast lexical heuristic here (an LLM-based judge is the Phase-2 upgrade) to
keep the verification gate cheap and deterministic.
"""

from __future__ import annotations

import os
from typing import Sequence

from openai import AsyncOpenAI

from app.models import Evidence
from app.providers.offline import _content  # reuse content-token overlap

_SYSTEM = (
    "You are a closed-book examiner. Answer the user's question using ONLY the "
    "evidence provided. Cite every claim with the source in square brackets like "
    "[doc_title]. If the evidence does not support an answer, reply exactly: "
    "'I don't have sufficient evidence to answer this confidently.' "
    "Do not use outside knowledge. Be concise — at most 3 sentences."
)


class OpenRouterGenerator:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.name = f"openrouter:{model}"
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

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
        # Reasoning models spend tokens on hidden reasoning; give them headroom and
        # retry once if the content comes back empty (a common free-tier hiccup).
        for max_tokens in (2000, 3000):
            resp = await self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                messages=messages,  # type: ignore[arg-type]
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        return "I don't have sufficient evidence to answer this confidently."

    async def judge_grounding(self, query: str, evidence: Sequence[Evidence]) -> tuple[float, bool]:
        if not evidence:
            return (0.0, False)
        q = _content(query)
        covered: set[str] = set()
        for e in evidence:
            covered |= (_content(e.text) & q)
        coverage = len(covered) / max(len(q), 1)
        return (round(min(1.0, 0.4 + coverage), 3), False)
