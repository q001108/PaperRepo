from __future__ import annotations

import hashlib
import math
import os
import re

from chromadb import Documents, EmbeddingFunction, Embeddings


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class HashEmbeddingFunction(EmbeddingFunction):
    """Small local embedding function for a runnable demo without API keys."""

    def __init__(self, dimensions: int, model_name: str = "") -> None:
        self.dimensions = dimensions
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(document) for document in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())

        for token in tokens:
            digest = hashlib.sha256(f"{self.model_name}:{token}".encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def get_embedding_function() -> EmbeddingFunction:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", "").strip()

    if provider in {"", "hash", "local_hash"}:
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
        return HashEmbeddingFunction(dimensions=dimensions, model_name=model_name)

    raise ValueError(
        f"Embedding provider '{provider}' is not configured in this demo. "
        "Set EMBEDDING_PROVIDER=hash, or extend src/embeddings.py with your provider."
    )
