"""
Hybrid retrieval: vector + full-text, fused, then reranked.

  1. Recall     cosine KNN over pgvector (HNSW) and ts_rank_cd over a
                generated tsvector, RETRIEVAL_POOL candidates each
  2. Fusion     reciprocal-rank fusion of the two ranked lists
  3. Precision  a cross-encoder re-scores the top RERANK_CANDIDATES

Retrieval used to be vector-only with a fixed cosine threshold. That missed
exact terms a question names (an acronym, "Table 4", a product code), because
those embed close to their neighbours, and the 0.5 threshold was a guess that
neither leg's scores can be compared against once fused.

Each stage degrades rather than fails: no query embedding means text-only
recall, and a reranker that is off or down means fused order.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from provena.core import config
from provena.core.db import get_connection
from provena.providers.embeddings import EmbeddingError, JinaEmbeddingClient
from provena.providers.reranker import JinaReranker

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """The chunk store could not be searched. Safe to show a user."""


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal-rank fusion.

    score(d) = sum over lists of 1 / (k + rank), rank starting at 1. Uses
    ranks rather than scores because cosine similarity and ts_rank_cd live on
    unrelated scales; adding them would let whichever is numerically larger
    decide. Ties break on first appearance, so the result is deterministic.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    position = 0
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = position
                position += 1
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


_COLUMNS = (
    "id", "document_id", "source_file", "chunk_id", "content", "start_offset",
    "end_offset", "heading_context", "section", "page_number", "metadata",
)


class Retriever:
    """Searches document_chunks and returns cited, ranked passages."""

    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or config.VECTOR_TABLE_NAME
        self.embedder = JinaEmbeddingClient()
        self.reranker = JinaReranker()
        logger.info(f"Initialized Retriever with table={self.table_name}")

    # ----------------------------------------------------------- recall ----

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        First-stage recall: both legs, fused. Best first.

        Each result carries `fusion` = {vector_rank, text_rank, rrf} so a bad
        answer can be traced to the leg that produced it.

        Raises RetrievalError if the database cannot be searched. It used to
        log and return [], which the router reported as "I couldn't find any
        relevant information" - a wrong answer rather than a visible failure.
        """
        pool = config.RETRIEVAL_POOL
        try:
            query_vector: Optional[List[float]] = self.embedder.embed_query(query)
        except EmbeddingError as exc:
            logger.warning(f"Query embedding failed, using full-text recall only: {exc}")
            query_vector = None

        scope = "(%(doc_ids)s::text[] IS NULL OR document_id::text = ANY(%(doc_ids)s::text[]))"
        params: Dict[str, Any] = {
            "q": query,
            "doc_ids": list(document_ids) if document_ids else None,
            "pool": pool,
        }

        # OR semantics for recall. plainto_tsquery ANDs every lexeme, so a
        # natural question ("what was total revenue in 2023") matched only
        # chunks containing all of them. ts_rank_cd still rewards covering more.
        legs = [f"""
            txt AS (
                SELECT id, ts_rank_cd(fts, tq.q) AS text_score
                FROM {self.table_name},
                     (SELECT replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery AS q) tq
                WHERE fts @@ tq.q AND {scope}
                ORDER BY text_score DESC, id
                LIMIT %(pool)s
            )"""]
        select_vec = "NULL::float AS similarity, NULL::bigint AS vector_rank"
        join_vec = ""
        if query_vector is not None:
            params["qv"] = query_vector
            legs.insert(0, f"""
            vec AS (
                SELECT id, embedding <=> %(qv)s::vector AS distance
                FROM {self.table_name}
                WHERE embedding IS NOT NULL AND {scope}
                ORDER BY embedding <=> %(qv)s::vector
                LIMIT %(pool)s
            ),
            vec_ranked AS (
                SELECT id, 1 - distance AS similarity,
                       row_number() OVER (ORDER BY distance, id) AS vector_rank
                FROM vec
            )""")
            select_vec = "v.similarity, v.vector_rank"
            join_vec = "LEFT JOIN vec_ranked v ON v.id = c.id"

        sql = f"""
            WITH {','.join(legs)},
            txt_ranked AS (
                SELECT id, row_number() OVER (ORDER BY text_score DESC, id) AS text_rank
                FROM txt
            )
            SELECT {', '.join('c.' + col for col in _COLUMNS)},
                   {select_vec}, t.text_rank
            FROM {self.table_name} c
            {join_vec}
            LEFT JOIN txt_ranked t ON t.id = c.id
            WHERE {'v.id IS NOT NULL OR ' if join_vec else ''}t.id IS NOT NULL
        """

        try:
            with get_connection(readonly=True) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                cur.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chunk search failed")
            raise RetrievalError("The document store could not be searched.") from exc

        by_id: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            record = dict(zip(_COLUMNS, row[: len(_COLUMNS)]))
            similarity, vector_rank, text_rank = row[len(_COLUMNS):]
            record["document_id"] = str(record["document_id"]) if record["document_id"] else None
            record["similarity"] = float(similarity) if similarity is not None else None
            record["fusion"] = {"vector_rank": vector_rank, "text_rank": text_rank}
            by_id[record["id"]] = record

        vector_order = sorted(
            (r for r in by_id.values() if r["fusion"]["vector_rank"] is not None),
            key=lambda r: r["fusion"]["vector_rank"],
        )
        text_order = sorted(
            (r for r in by_id.values() if r["fusion"]["text_rank"] is not None),
            key=lambda r: r["fusion"]["text_rank"],
        )

        fused = rrf_fuse(
            [[r["id"] for r in vector_order], [r["id"] for r in text_order]],
            k=config.RRF_K,
        )

        results = []
        for doc_id, rrf in fused[: limit or len(fused)]:
            record = by_id[doc_id]
            record["fusion"]["rrf"] = round(rrf, 6)
            # Until a reranker scores it, the displayed score is cosine
            # similarity when the vector leg found it.
            record["score"] = record["similarity"]
            record["score_kind"] = "cosine" if record["similarity"] is not None else "text"
            results.append(record)
        return results

    # -------------------------------------------------------- precision ----

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Recall, fuse, rerank. Returns (results best first, reranked?).

        With reranking, `score` is the cross-encoder's relevance and
        `score_kind` is "rerank". Without it the fused order stands.
        """
        top_k = top_k or config.RAG_TOP_K
        candidates = self.search(
            query, limit=config.RERANK_CANDIDATES, document_ids=document_ids
        )
        if not candidates:
            return [], False

        passages = [
            f"{c['heading_context']}\n\n{c['content']}" if c.get("heading_context") else c["content"]
            for c in candidates
        ]
        ranked = self.reranker.rerank(query, passages, top_n=top_k)
        if ranked is None:
            return candidates[:top_k], False

        out = []
        for index, relevance in ranked:
            record = candidates[index]
            record["fusion"]["fused_rank"] = index + 1
            record["score"] = round(relevance, 4)
            record["score_kind"] = "rerank"
            out.append(record)
        return out, True

    @staticmethod
    def build_context(results: List[Dict[str, Any]]) -> str:
        """Context block for the answer prompt, numbered to match `sources`."""
        if not results:
            return "No relevant context found."

        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Source {i}]"
            if result.get("page_number"):
                header += f" (page {result['page_number']})"
            if result.get("heading_context"):
                header += f" {result['heading_context']}"
            parts.append(f"{header}\n{result['content']}")

        return "\n\n---\n\n".join(parts)
