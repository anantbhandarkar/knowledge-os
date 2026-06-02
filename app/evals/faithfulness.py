"""Pure, deterministic faithfulness metric.

Faithfulness here is a cheap lexical-grounding proxy: the fraction of the answer's
content words that actually appear somewhere in the retrieved evidence. It is fully
offline and side-effect free so it stays stable across test runs.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been being of for to in on at by with and or "
    "what which who whom how when where why do does did this that these those it "
    "its as from into about can could should would will i you we they he she".split()
)


def _content_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens of length > 2, minus a small stopword set."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOP]


def faithfulness(answer: str, evidence_texts: list[str]) -> float:
    """Fraction of the answer's content words present in the concatenated evidence.

    Returns a value in [0, 1]; 0 when the answer has no content words.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    evidence_vocab = set(_content_tokens(" ".join(evidence_texts)))
    if not evidence_vocab:
        return 0.0
    grounded = sum(1 for t in answer_tokens if t in evidence_vocab)
    return grounded / len(answer_tokens)
