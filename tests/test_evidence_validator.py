from src.evidence_validator import filter_valid_evidence
from src.schemas import RetrievedChunk


def test_invalid_repo_file_path_is_dropped_from_valid_evidence():
    chunks = [
        RetrievedChunk(
            content="valid readme",
            metadata={
                "dataset_id": "dataset-1",
                "source_type": "repo",
                "repo_url": "https://github.com/example/repo",
                "file_path": "README.md",
                "line_start": 1,
                "line_end": 1,
            },
            score=0.9,
            source_type="repo",
        ),
        RetrievedChunk(
            content="polluted file",
            metadata={
                "dataset_id": "dataset-1",
                "source_type": "repo",
                "repo_url": "https://github.com/example/repo",
                "file_path": "models/timm_models/util/layers/gather_excite.py",
                "line_start": 1,
                "line_end": 1,
            },
            score=0.9,
            source_type="repo",
        ),
    ]

    valid = filter_valid_evidence(
        retrieved_chunks=chunks,
        dataset_id="dataset-1",
        repo_url="https://github.com/example/repo",
        scanned_file_paths={"README.md"},
    )

    assert len(valid) == 1
    assert valid[0].metadata["file_path"] == "README.md"


def test_wrong_repo_url_is_dropped_from_valid_evidence():
    chunks = [
        RetrievedChunk(
            content="wrong repo",
            metadata={
                "dataset_id": "dataset-1",
                "source_type": "repo",
                "repo_url": "https://github.com/other/repo",
                "file_path": "README.md",
                "line_start": 1,
                "line_end": 1,
            },
            score=0.9,
            source_type="repo",
        )
    ]

    valid = filter_valid_evidence(
        retrieved_chunks=chunks,
        dataset_id="dataset-1",
        repo_url="https://github.com/example/repo",
        scanned_file_paths={"README.md"},
    )

    assert valid == []
