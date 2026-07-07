from __future__ import annotations

import logging
from pathlib import Path
import shutil
import tempfile
from collections.abc import Iterator
from urllib.parse import urlparse

from git import GitCommandError, Repo
from pydantic import HttpUrl, TypeAdapter, ValidationError

from src.dataset import normalize_github_url
from src.schemas import RepoChunk, RepoScanResult


logger = logging.getLogger(__name__)
http_url_adapter = TypeAdapter(HttpUrl)
MAX_REPO_FILES = 200
MAX_FILE_SIZE_BYTES = 512 * 1024
MAX_TOTAL_SIZE_BYTES = 5 * 1024 * 1024
MAX_LINES_PER_CHUNK = 120
REPO_CACHE_DIR = Path(".repos")
ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".json"}
KEY_FILENAMES = {"readme", "requirements.txt", "environment.yml", "dockerfile"}


def validate_github_url(repo_url: str) -> str:
    """Validate that a URL points to a public GitHub repository shape."""
    try:
        parsed_url = http_url_adapter.validate_python(repo_url.strip())
    except ValidationError as exc:
        raise ValueError("Please enter a valid GitHub repository URL.") from exc

    parsed = urlparse(str(parsed_url))
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]

    if parsed.netloc.lower() != "github.com" or len(path_parts) < 2:
        raise ValueError("Please enter a URL like https://github.com/owner/repository.")

    normalized_url = normalize_github_url(str(parsed_url))
    logger.info("GitHub repository URL accepted: %s", normalized_url)
    return normalized_url


def is_scannable_file(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    return name in KEY_FILENAMES or stem == "readme" or path.suffix.lower() in ALLOWED_SUFFIXES


def is_key_file_path(file_path: str) -> bool:
    return is_scannable_file(Path(file_path)) and (
        Path(file_path).name.lower() in KEY_FILENAMES or Path(file_path).stem.lower() == "readme"
    )


def clone_repository(repo_url: str, destination_root: Path = REPO_CACHE_DIR) -> Path:
    """Clone a public GitHub repository without executing repository code."""
    validated_url = validate_github_url(repo_url)
    destination_root.mkdir(exist_ok=True)
    destination_path = Path(
        tempfile.mkdtemp(prefix="repo-", dir=destination_root)
    )

    try:
        Repo.clone_from(
            validated_url,
            destination_path,
            depth=1,
            multi_options=["--single-branch"],
            env={"GIT_TERMINAL_PROMPT": "0"},
        )
    except GitCommandError as exc:
        shutil.rmtree(destination_path, ignore_errors=True)
        raise ValueError("Unable to clone the GitHub repository. Confirm it is public and reachable.") from exc
    except Exception:
        shutil.rmtree(destination_path, ignore_errors=True)
        raise

    logger.info("Cloned repository %s to %s", validated_url, destination_path)
    return destination_path


def get_repo_commit_hash(repo_path: Path) -> str | None:
    try:
        return Repo(repo_path).head.commit.hexsha
    except Exception as exc:
        logger.warning("Unable to read repository commit hash for %s: %s", repo_path, exc)
        return None


def _iter_candidate_files(repo_path: Path) -> Iterator[Path]:
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.relative_to(repo_path).parts:
            continue
        if is_scannable_file(path):
            yield path


def _language_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "Python"
    if suffix in {".yaml", ".yml"}:
        return "YAML"
    if suffix == ".json":
        return "JSON"
    if path.name.lower() == "dockerfile":
        return "Dockerfile"
    return None


def scan_repository_path(
    repo_path: Path,
    repo_url: str | None = None,
    max_files: int = MAX_REPO_FILES,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_total_size_bytes: int = MAX_TOTAL_SIZE_BYTES,
) -> list[RepoChunk]:
    """Statically scan allowed repository files into line-aware chunks."""
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    chunks: list[RepoChunk] = []
    scanned_files = 0
    scanned_bytes = 0

    for file_path in _iter_candidate_files(repo_path):
        if scanned_files >= max_files:
            logger.warning("Repository scan stopped at max file count: %d", max_files)
            break

        file_size = file_path.stat().st_size
        if file_size > max_file_size_bytes:
            logger.warning("Skipping large file: %s (%d bytes)", file_path, file_size)
            continue

        if scanned_bytes + file_size > max_total_size_bytes:
            logger.warning("Repository scan stopped at max total size: %d bytes", max_total_size_bytes)
            break

        relative_path = file_path.relative_to(repo_path).as_posix()
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if not lines:
            continue

        scanned_files += 1
        scanned_bytes += file_size

        for start in range(0, len(lines), MAX_LINES_PER_CHUNK):
            end = min(start + MAX_LINES_PER_CHUNK, len(lines))
            chunk_lines = lines[start:end]
            chunks.append(
                RepoChunk(
                    chunk_id=f"repo-{relative_path.replace('/', '-')}-{start + 1}-{end}",
                    content="\n".join(chunk_lines),
                    file_path=relative_path,
                    line_start=start + 1,
                    line_end=end,
                    repo_url=repo_url,
                    language=_language_for(file_path),
                )
            )

    logger.info(
        "Scanned repository %s: %d files, %d chunks, %d bytes",
        repo_path,
        scanned_files,
        len(chunks),
        scanned_bytes,
    )
    return chunks


def scan_repository(repo_url: str) -> list[RepoChunk]:
    """Clone and statically scan a public GitHub repository."""
    return scan_repository_with_metadata(repo_url).chunks


def scan_repository_with_metadata(repo_url: str) -> RepoScanResult:
    """Clone and statically scan a public GitHub repository with repository metadata."""
    normalized_url = validate_github_url(repo_url)
    repo_path = clone_repository(normalized_url)
    commit_hash = get_repo_commit_hash(repo_path)
    chunks = scan_repository_path(repo_path, repo_url=normalized_url)
    scanned_files = sorted({chunk.file_path for chunk in chunks})
    return RepoScanResult(
        chunks=chunks,
        repo_url=normalized_url,
        commit_hash=commit_hash,
        scanned_files=scanned_files,
    )
