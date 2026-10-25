"""
Cross-encoder reranking via Jina's rerank API.

First-stage retrieval (vector + full-text, fused) is tuned for recall: it
scores the query and each chunk independently. A cross-encoder reads the query
and chunk together, which is far better at precision, and is too slow to run
over the whole corpus - so it re-scores only the fused candidate pool.

Reuses JINA_API_KEY, so it adds no vendor. Reranking is an improvement, never
a dependency: every failure path returns None and the caller keeps the fused
order.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import requests

from provena.core import config

logger = logging.getLogger(__name__)


class JinaReranker:
    """Scores (query, passage) pairs with a cross-encoder."""

    def __init__(self) -> None:
        self.api_key = config.JINA_API_KEY
        self.api_url = config.RERANK_URL
        self.model = config.RERANK_MODEL
        self.enabled = bool(config.RERANK_ENABLED and self.api_key)
        if not self.enabled:
            logger.info("Reranking disabled; retrieval keeps fused order")

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        top_n: int,
        max_chars: int = 4000,
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Returns [(passage_index, relevance_score)] best first, or None when
        reranking is disabled or failed.
        """
        if not self.enabled or not passages:
            return None

        payload = {
            "model": self.model,
            "query": query,
            "documents": [p[:max_chars] for p in passages],
            "top_n": min(top_n, len(passages)),
            "return_documents": False,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()["results"]
            ranked = [(int(r["index"]), float(r["relevance_score"])) for r in results]
        except (requests.exceptions.RequestException, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Rerank failed, keeping fused order: {exc}")
            return None

        if any(not 0 <= i < len(passages) for i, _ in ranked):
            logger.warning("Rerank returned an out-of-range index, keeping fused order")
            return None
        return ranked
