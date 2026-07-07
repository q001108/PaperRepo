from __future__ import annotations

import os
import uuid

import chromadb
from pydantic import BaseModel, Field

from src.embeddings import get_embedding_function
from src.schemas import PaperChunk, RepoChunk


class IndexBuildResult(BaseModel):
    dataset_id: str
    collection_name: str
    document_count: int = Field(..., ge=0)


def _collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", "").strip() or "paperrepo_evidence"


def _chroma_path() -> str:
    return os.getenv("CHROMA_PATH", "").strip() or ".chroma"


def get_collection():
    client = chromadb.PersistentClient(path=_chroma_path())
    return client.get_or_create_collection(
        name=_collection_name(),
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def clear_dataset_index(dataset_id: str) -> None:
    """Remove all chunks for a dataset before rebuilding its index."""
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty.")
    collection = get_collection()
    existing = collection.get(where={"dataset_id": dataset_id}, include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def _clean_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def build_index(
    paper_chunks: list[PaperChunk],
    repo_chunks: list[RepoChunk],
    dataset_id: str | None = None,
    repo_url: str = "",
    repo_commit_hash: str | None = None,
    clear_existing: bool = True,
) -> IndexBuildResult:
    """Write paper and repo chunks into the same Chroma collection."""
    collection = get_collection()
    current_dataset_id = dataset_id or uuid.uuid4().hex

    if clear_existing:
        clear_dataset_index(current_dataset_id)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk in paper_chunks:
        ids.append(f"{current_dataset_id}:{chunk.chunk_id}")
        documents.append(chunk.content)
        metadatas.append(
            _clean_metadata(
                {
                    "dataset_id": current_dataset_id,
                    "chunk_id": chunk.chunk_id,
                    "source_type": chunk.source_type,
                    "repo_url": repo_url,
                    "repo_commit_hash": repo_commit_hash or "",
                    "page_num": chunk.page_num,
                    "section_title": chunk.section_title or "",
                    "source_file": chunk.source_file or "",
                }
            )
        )

    for chunk in repo_chunks:
        ids.append(f"{current_dataset_id}:{chunk.chunk_id}")
        documents.append(chunk.content)
        metadatas.append(
            _clean_metadata(
                {
                    "dataset_id": current_dataset_id,
                    "chunk_id": chunk.chunk_id,
                    "source_type": chunk.source_type,
                    "file_path": chunk.file_path,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "repo_url": repo_url or (str(chunk.repo_url) if chunk.repo_url else ""),
                    "repo_commit_hash": repo_commit_hash or "",
                    "language": chunk.language or "",
                }
            )
        )

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return IndexBuildResult(
        dataset_id=current_dataset_id,
        collection_name=_collection_name(),
        document_count=len(ids),
    )
