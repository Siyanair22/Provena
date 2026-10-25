"""
Query decomposition for the RAG route.

A compound question ("what is the churn rate and how does it compare with
last year's target?") retrieved as one query gets passages about whichever
half embeds louder. Splitting it lets each part retrieve on its own, and gives
each part its own coverage verdict, so the answer can say which half the
documents do not cover instead of guessing at it.

Only the split itself uses the model. The gate that decides whether to try,
and the verdicts, are pure functions: a verdict is a statement about what
retrieval found, and it would be circular to let the model grade that.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Sequence

from provena.core import config

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3

Verdict = Literal["ANSWERED", "PARTIAL", "NOT_FOUND"]

_COMPOUND_SIGNALS = [
    r"\bcompare\b", r"\bcompared (to|with)\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bdifference between\b", r"\bas well as\b", r"\bboth\b",
]
# "and" joining two clauses that each ask something, not "revenue and profit".
_CLAUSE_AND = re.compile(
    r"\band\s+(what|how|why|when|where|which|who|is|are|does|do|did|can)\b", re.I
)


def should_decompose(query: str) -> bool:
    """
    Cheap gate in front of the model call.

    Most questions are simple, and on the free tier every call counts (8000
    tokens per minute, and a HYBRID query already makes several), so the
    split is only attempted when the wording suggests more than one question.
    """
    q = query.strip().lower()
    if len(q.split()) < 6:
        return False
    if q.count("?") >= 2:
        return True
    if _CLAUSE_AND.search(q):
        return True
    return any(re.search(p, q) for p in _COMPOUND_SIGNALS)


DECOMPOSE_PROMPT = f"""Split the user's question into the separate questions it contains.

Rules:
- Return between 1 and {MAX_SUB_QUESTIONS} questions.
- Each must stand alone: repeat the subject instead of writing "it" or "they".
- Do not add questions the user did not ask.
- If it is already a single question, return it unchanged as the only item.

Return JSON only: {{"questions": ["...", "..."]}}"""


def parse_sub_questions(raw: str, original: str) -> List[str]:
    """Validate the model's split; anything unusable means "don't split"."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return [original]

    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        return [original]

    cleaned: List[str] = []
    for q in questions:
        if isinstance(q, str) and 3 <= len(q.strip()) <= 500 and q.strip() not in cleaned:
            cleaned.append(q.strip())
    return cleaned[:MAX_SUB_QUESTIONS] or [original]


def decompose(llm: Any, query: str) -> List[str]:
    """Sub-questions for a query, or [query] if it is simple or the split fails."""
    if not should_decompose(query):
        return [query]
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - a failed split is not a failed answer
        logger.warning(f"Decomposition failed, retrieving the question whole: {exc}")
        return [query]
    return parse_sub_questions(raw, query)


def verdict(results: Sequence[Dict[str, Any]], reranked: bool) -> Verdict:
    """
    What retrieval found for one sub-question.

    With reranking the cross-encoder's relevance is the evidence: a passage
    at or above RERANK_ANSWERED_SCORE addresses the question, weaker ones only
    touch it. Without reranking there is no calibrated score to lean on, so
    anything retrieved is PARTIAL rather than claimed as an answer.
    """
    if not results:
        return "NOT_FOUND"
    if not reranked:
        return "PARTIAL"
    best = max((r.get("score") or 0.0) for r in results)
    return "ANSWERED" if best >= config.RERANK_ANSWERED_SCORE else "PARTIAL"
