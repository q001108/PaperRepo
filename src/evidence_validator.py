from __future__ import annotations

from src.dataset import normalize_github_url
from src.schemas import RetrievedChunk


EXCLUDED_PAPER_SECTIONS = {"references", "参考文献"}


def filter_valid_evidence(
    retrieved_chunks: list[RetrievedChunk],
    dataset_id: str,
    repo_url: str,
    scanned_file_paths: set[str],
) -> list[RetrievedChunk]:
    """Drop chunks that do not belong to the current dataset and scanned repository."""
    normalized_repo_url = normalize_github_url(repo_url)
    valid_chunks: list[RetrievedChunk] = []

    for chunk in retrieved_chunks:
        metadata = chunk.metadata
        if metadata.get("dataset_id") != dataset_id:
            continue

        if chunk.source_type == "paper":
            section_title = str(metadata.get("section_title", "")).strip().lower()
            if section_title in EXCLUDED_PAPER_SECTIONS:
                continue
            valid_chunks.append(chunk)
            continue

        if chunk.source_type != "repo":
            continue

        chunk_repo_url = str(metadata.get("repo_url", ""))
        if normalize_github_url(chunk_repo_url) != normalized_repo_url:
            continue

        file_path = str(metadata.get("file_path", ""))
        if file_path not in scanned_file_paths:
            continue

        valid_chunks.append(chunk)

    return valid_chunks
