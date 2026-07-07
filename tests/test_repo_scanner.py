from pathlib import Path

from src.repo_scanner import is_key_file_path, scan_repository_path


def test_scan_repository_path_reads_only_allowed_static_files(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo_path / "requirements.txt").write_text("streamlit\n", encoding="utf-8")
    (repo_path / "notes.txt").write_text("ignore me\n", encoding="utf-8")
    src_dir = repo_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('static only')\n", encoding="utf-8")

    chunks = scan_repository_path(repo_path)
    file_paths = {chunk.file_path for chunk in chunks}

    assert "README.md" in file_paths
    assert "requirements.txt" in file_paths
    assert "src/main.py" in file_paths
    assert "notes.txt" not in file_paths
    assert all(chunk.source_type == "repo" for chunk in chunks)
    assert all(chunk.line_start <= chunk.line_end for chunk in chunks)
    assert is_key_file_path("README.md")


def test_scan_repository_path_respects_file_size_limit(tmp_path: Path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (repo_path / "large.py").write_text("x" * 128, encoding="utf-8")

    chunks = scan_repository_path(repo_path, max_file_size_bytes=16)
    file_paths = {chunk.file_path for chunk in chunks}

    assert "small.py" in file_paths
    assert "large.py" not in file_paths
