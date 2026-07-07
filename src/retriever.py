from __future__ import annotations

from typing import Literal

from src.indexer import get_collection
from src.schemas import RetrievedChunk


SourceFilter = Literal["paper", "repo", "both"]


def _where_filter(dataset_id: str, source_filter: SourceFilter):
    if source_filter == "both":
        return {"dataset_id": dataset_id}
    return {
        "$and": [
            {"dataset_id": dataset_id},
            {"source_type": source_filter},
        ]
    }


def retrieve_evidence(
    question: str,
    dataset_id: str,
    source_filter: SourceFilter = "both",
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve raw chunk content and full metadata from Chroma."""
    if source_filter not in {"paper", "repo", "both"}:
        raise ValueError("source_filter must be one of: paper, repo, both.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty.")

    collection = get_collection()
    total_count = collection.count()
    if total_count == 0:
        return []

    result_count = max(1, min(top_k, total_count))
    results = collection.query(
        query_texts=[question],
        n_results=result_count,
        where=_where_filter(dataset_id, source_filter),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        source_type = metadata.get("source_type")
        if source_type not in {"paper", "repo"}:
            continue
        retrieved.append(
            RetrievedChunk(
                content=content,
                metadata=metadata,
                score=max(0.0, 1.0 - float(distance)),
                source_type=source_type,
            )
        )
    return retrieved
