"""Golden evaluation set grounded in the demo corpus facts.

Each entry pairs a query with an expected answer substring and the expected gate
decision. Facts encoded here (must match the seeded demo corpus):
  - contractor JWT tokens expire after 4 hours
  - standard employee tokens expire after 8 hours
  - audit logs are retained for 2 years
  - backups are retained for 35 days
  - on-call engineers must acknowledge within 5 minutes

The final item is an unanswerable query with no supporting evidence; the gate is
expected to abstain rather than fabricate an answer.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class GoldenItem(TypedDict):
    query: str
    expect_substring: str
    expect_decision: Literal["pass", "abstain"]


GOLDEN: list[GoldenItem] = [
    {
        "query": "How long until contractor JWT tokens expire?",
        "expect_substring": "4 hours",
        "expect_decision": "pass",
    },
    {
        "query": "How long do standard employee tokens last?",
        "expect_substring": "8 hours",
        "expect_decision": "pass",
    },
    {
        "query": "How long are audit logs retained?",
        "expect_substring": "2 years",
        "expect_decision": "pass",
    },
    {
        "query": "What is the backup retention period?",
        "expect_substring": "35 days",
        "expect_decision": "pass",
    },
    {
        "query": "Within how long must on-call engineers acknowledge an incident?",
        "expect_substring": "5 minutes",
        "expect_decision": "pass",
    },
    {
        "query": "What is the company's policy on remote work in Antarctica?",
        "expect_substring": "",
        "expect_decision": "abstain",
    },
]
