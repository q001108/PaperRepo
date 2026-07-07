from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def normalize_github_url(repo_url: str) -> str:
    parsed = urlparse(repo_url.strip())
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    path_parts = [part for part in path.split("/") if part]
    if len(path_parts) >= 2:
        path = "/".join(path_parts[:2])
    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=f"/{path}",
        params="",
        query="",
        fragment="",
    )
    return urlunparse(normalized).rstrip("/")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_id(pdf_hash: str, normalized_repo_url: str, commit_hash: str | None) -> str:
    raw = "|".join([pdf_hash, normalized_repo_url, commit_hash or "unknown"])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
