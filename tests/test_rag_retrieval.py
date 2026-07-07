from pathlib import Path

from src.indexer import build_index
from src.retriever import retrieve_evidence
from src.schemas import PaperChunk, RepoChunk


def test_chroma_retrieval_supports_source_filters(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "64")

    paper_chunks = [
        PaperChunk(
            chunk_id="paper-1",
            content="Attention retrieval evidence appears in the paper.",
            page_num=2,
            section_title="Method",
            source_file="paper.pdf",
        )
    ]
    repo_chunks = [
        RepoChunk(
            chunk_id="repo-1",
            content="def retrieval_pipeline(): return 'repo evidence'",
            file_path="src/pipeline.py",
            line_start=10,
            line_end=11,
            language="Python",
            repo_url="https://github.com/example/project",
        )
    ]

    index_result = build_index(
        paper_chunks,
        repo_chunks,
        dataset_id="test-dataset",
        repo_url="https://github.com/example/project",
    )

    assert index_result.document_count == 2

    paper_results = retrieve_evidence(
        "attention paper evidence",
        dataset_id=index_result.dataset_id,
        source_filter="paper",
        top_k=5,
    )
    repo_results = retrieve_evidence(
        "retrieval pipeline repo evidence",
        dataset_id=index_result.dataset_id,
        source_filter="repo",
        top_k=5,
    )
    both_results = retrieve_evidence(
        "retrieval evidence",
        dataset_id=index_result.dataset_id,
        source_filter="both",
        top_k=5,
    )

    assert paper_results
    assert paper_results[0].source_type == "paper"
    assert paper_results[0].metadata["page_num"] == 2
    assert "paper" in paper_results[0].content.lower()

    assert repo_results
    assert repo_results[0].source_type == "repo"
    assert repo_results[0].metadata["file_path"] == "src/pipeline.py"
    assert repo_results[0].metadata["line_start"] == 10
    assert "repo evidence" in repo_results[0].content

    assert {result.source_type for result in both_results} == {"paper", "repo"}


def test_chroma_retrieval_is_isolated_by_dataset_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "isolation_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "64")

    build_index(
        paper_chunks=[],
        repo_chunks=[
            RepoChunk(
                chunk_id="repo-a",
                content="alpha repository evidence",
                file_path="README.md",
                line_start=1,
                line_end=1,
                repo_url="https://github.com/example/alpha",
            )
        ],
        dataset_id="dataset-alpha",
        repo_url="https://github.com/example/alpha",
    )
    build_index(
        paper_chunks=[],
        repo_chunks=[
            RepoChunk(
                chunk_id="repo-b",
                content="beta repository evidence",
                file_path="README.md",
                line_start=1,
                line_end=1,
                repo_url="https://github.com/example/beta",
            )
        ],
        dataset_id="dataset-beta",
        repo_url="https://github.com/example/beta",
    )

    alpha_results = retrieve_evidence(
        "beta repository evidence",
        dataset_id="dataset-alpha",
        source_filter="repo",
        top_k=5,
    )

    assert alpha_results
    assert all(result.metadata["dataset_id"] == "dataset-alpha" for result in alpha_results)
    assert all("beta repository evidence" not in result.content for result in alpha_results)
