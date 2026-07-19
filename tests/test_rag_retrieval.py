from pathlib import Path

from src.indexer import build_index
from src.retriever import _ranking_score, retrieve_evidence
from src.schemas import RetrievedChunk
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


def test_both_retrieval_queries_each_source_even_with_small_top_k(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "balanced_both_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "64")

    paper_chunks = [
        PaperChunk(
            chunk_id=f"paper-{index}",
            content=f"shared retrieval method evidence paper detail {index}",
            page_num=index,
            section_title="Method",
            source_file="paper.pdf",
        )
        for index in range(1, 4)
    ]
    repo_chunks = [
        RepoChunk(
            chunk_id="repo-1",
            content="shared retrieval method evidence repository implementation",
            file_path="README.md",
            line_start=1,
            line_end=2,
            repo_url="https://github.com/example/project",
        )
    ]

    index_result = build_index(
        paper_chunks,
        repo_chunks,
        dataset_id="balanced-dataset",
        repo_url="https://github.com/example/project",
    )

    both_results = retrieve_evidence(
        "shared retrieval method evidence",
        dataset_id=index_result.dataset_id,
        source_filter="both",
        top_k=1,
    )

    assert "paper" in {result.source_type for result in both_results}
    assert "repo" in {result.source_type for result in both_results}


def test_repo_reranking_prefers_project_implementation_files():
    readme = RetrievedChunk(
        content="official implementation",
        metadata={"source_type": "repo", "file_path": "README.md"},
        score=0.4,
        source_type="repo",
    )
    model = RetrievedChunk(
        content="model implementation",
        metadata={"source_type": "repo", "file_path": "models/mlic.py"},
        score=0.4,
        source_type="repo",
    )
    script = RetrievedChunk(
        content="dataset preprocessing",
        metadata={"source_type": "repo", "file_path": "scripts/mscoco.py"},
        score=0.4,
        source_type="repo",
    )
    vendored = RetrievedChunk(
        content="third-party backbone",
        metadata={"source_type": "repo", "file_path": "models/timm_models/vision_transformer.py"},
        score=0.4,
        source_type="repo",
    )

    assert _ranking_score(readme) > _ranking_score(script)
    assert _ranking_score(model) > _ranking_score(vendored)


def test_repo_method_questions_demote_vendored_helpers(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "method_repo_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "64")

    repo_chunks = [
        RepoChunk(
            chunk_id="repo-readme",
            content=(
                "This is an official PyTorch implementation of "
                "Semantic-Guided Representation Enhancement. "
                "python train.py --model mlic"
            ),
            file_path="README.md",
            line_start=1,
            line_end=4,
            repo_url="https://github.com/example/sgre",
        ),
        RepoChunk(
            chunk_id="repo-model",
            content=(
                "class MLIC(nn.Module):\n"
                "self.attention = LowRankBilinearAttention(feat_dim, text_dim, 1024)\n"
                "alpha = self.attention(x, embeddings, self.cfg.tau)"
            ),
            file_path="models/mlic.py",
            line_start=205,
            line_end=260,
            language="Python",
            repo_url="https://github.com/example/sgre",
        ),
        RepoChunk(
            chunk_id="repo-helper",
            content="def to_2tuple(x): return tuple([x, x])\nclass Format: pass\nimplementation method model",
            file_path="models/timm_models/util/helpers.py",
            line_start=1,
            line_end=3,
            language="Python",
            repo_url="https://github.com/example/sgre",
        ),
    ]

    index_result = build_index(
        paper_chunks=[],
        repo_chunks=repo_chunks,
        dataset_id="method-repo-dataset",
        repo_url="https://github.com/example/sgre",
    )

    results = retrieve_evidence(
        "这个仓库实现了论文的哪些方法",
        dataset_id=index_result.dataset_id,
        source_filter="repo",
        top_k=2,
    )

    assert [result.metadata["file_path"] for result in results] == ["models/mlic.py", "README.md"]


def test_paper_method_questions_prefer_method_sections():
    overview = RetrievedChunk(
        content=(
            "We propose a semantic-guided representation enhancement framework. "
            "It consists of an inter-modal attention module and an intra-modal attention module."
        ),
        metadata={"source_type": "paper", "section_title": "A. Overview"},
        score=0.2,
        source_type="paper",
    )
    ablation = RetrievedChunk(
        content="The ablation study reports the effects of modules on model performance.",
        metadata={"source_type": "paper", "section_title": "D. Ablation Study"},
        score=0.35,
        source_type="paper",
    )

    assert _ranking_score(overview) > _ranking_score(ablation)
