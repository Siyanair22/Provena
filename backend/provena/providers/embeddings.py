"""
Shared Jina Embeddings Module

Single canonical JinaEmbeddingClient used by all modules.
Based on the batch-capable version from chunker.py.
"""

import logging
from typing import List

import requests

from provena.core import config

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The embedding API failed or returned something unusable."""


class JinaEmbeddingClient:
    """
    Jina AI embeddings client for generating vectors.
    Supports single and batch embeddings for efficiency.
    """

    def __init__(self):
        self.api_key = config.JINA_API_KEY
        if not self.api_key:
            raise ValueError("JINA_API_KEY is required (set it in the repo-root .env)")

        self.api_url = config.JINA_API_URL
        self.model = config.JINA_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS
        self.query_task = config.JINA_QUERY_TASK
        self.passage_task = config.JINA_PASSAGE_TASK
        self.batch_size = config.JINA_BATCH_SIZE

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Initialized JinaEmbeddingClient with model={self.model}, "
            f"dimensions={self.dimensions}"
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query."""
        return self.embed_batch([text], task=self.query_task)[0]

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks for storage."""
        return self.embed_batch(texts, task=self.passage_task)

    def embed_batch(
        self,
        texts: List[str],
        task: str,
        max_chars: int = 12000,
    ) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in batched API calls.

        Jina v3 is asymmetric: queries and passages are embedded with different
        task adapters (retrieval.query / retrieval.passage). Both used to share
        "text-matching", which is tuned for symmetric similarity and ranks a
        short question against long passages worse.

        Raises EmbeddingError rather than returning empty vectors. A failed
        batch used to yield [] per text, and upsert_chunks then skipped those
        chunks without failing the document, so a PDF could finish "completed"
        with half its text unsearchable.
        """
        if not texts:
            return []

        truncated_texts = [text[:max_chars] for text in texts]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(truncated_texts), self.batch_size):
            batch = truncated_texts[i : i + self.batch_size]

            payload = {
                "model": self.model,
                "task": task,
                "dimensions": self.dimensions,
                "input": batch,
            }

            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            except requests.exceptions.RequestException as e:
                raise EmbeddingError(f"Jina API request failed: {e}") from e
            except (KeyError, IndexError, ValueError) as e:
                raise EmbeddingError(f"Unexpected response format from Jina API: {e}") from e

            if len(batch_embeddings) != len(batch):
                raise EmbeddingError(
                    f"Jina returned {len(batch_embeddings)} vectors for {len(batch)} texts"
                )
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Embedded batch {i // self.batch_size + 1}, {len(batch)} texts")

        return all_embeddings
