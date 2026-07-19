from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

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


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Embedding function backed by a local sentence-transformers model."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        if not model_name:
            raise ValueError("EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=sentence_transformers.")
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def __call__(self, input: Documents) -> Embeddings:
        model = self._load_model()
        embeddings = model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ValueError(
                    "sentence-transformers is not installed. Run `pip install -r requirements.txt` "
                    "after adding EMBEDDING_PROVIDER=sentence_transformers."
                ) from exc

            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model


def get_embedding_function() -> EmbeddingFunction:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", "").strip()

    if provider in {"", "hash", "local_hash"}:
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
        return HashEmbeddingFunction(dimensions=dimensions, model_name=model_name)

    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        device = os.getenv("SENTENCE_TRANSFORMERS_DEVICE", "").strip() or None
        return SentenceTransformerEmbeddingFunction(model_name=model_name, device=device)

    raise ValueError(
        f"Embedding provider '{provider}' is not configured in this demo. "
        "Set EMBEDDING_PROVIDER=hash or EMBEDDING_PROVIDER=sentence_transformers."
    )
