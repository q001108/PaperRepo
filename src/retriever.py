from __future__ import annotations

from typing import Literal

from src.indexer import get_collection
from src.schemas import RetrievedChunk


SourceFilter = Literal["paper", "repo", "both"]

REPO_QUERY_EXPANSION = (
    " official implementation README model train framework architecture method module attention "
    "Semantic-Guided Representation Enhancement SGRE mlic inter-modal intra-modal "
    "repository implementation paper method framework model module "
    "仓库 实现 论文 方法 框架 模型 模块 注意力"
)
PAPER_METHOD_QUERY_EXPANSION = (
    " proposed method framework overview inter-modal attention intra-modal attention "
    "joint learning semantic-guided representation enhancement module architecture "
    "论文 提出 方法 框架 模块 注意力 概述 联合学习"
)

IMPORTANT_REPO_FILES = {
    "readme.md": 0.45,
    "readme": 0.45,
    "models/mlic.py": 0.70,
    "train.py": 0.28,
    "models/factory.py": 0.20,
    "embedding.py": 0.16,
    "evaluate.py": 0.10,
    "infer.py": 0.10,
    "lib/aslloss.py": 0.08,
}

NOISY_REPO_PATH_PENALTIES = {
    "models/timm_models/": 0.38,
    "models/cvt_models/": 0.34,
    "models/torch_models/": 0.34,
    "models/tresnet/": 0.34,
    "scripts/": 0.22,
    "lib/metrics.py": 0.30,
    "lib/util.py": 0.16,
}

GENERIC_REPO_FILE_PENALTIES = {
    "__init__.py": 0.18,
    "helpers.py": 0.28,
    "helper.py": 0.24,
    "utils.py": 0.20,
    "util.py": 0.18,
}

CORE_REPO_CONTENT_MARKERS = {
    "official pytorch implementation": 0.18,
    "semantic-guided representation enhancement": 0.16,
    "class mlic": 0.18,
    "lowrankbilinearattention": 0.18,
    "transformerencoder": 0.10,
    "alpha = self.attention": 0.12,
    "register_model": 0.08,
    "python train.py --model mlic": 0.14,
    "create_model": 0.06,
}

IMPLEMENTATION_METHOD_QUERY_MARKERS = {
    "implement",
    "implementation",
    "method",
    "methods",
    "framework",
    "architecture",
    "model",
    "module",
    "paper",
    "实现",
    "方法",
    "框架",
    "模型",
    "模块",
    "论文",
}

METHOD_SECTIONS = {
    "abstract": 0.05,
    "method": 0.12,
    "methods": 0.12,
    "methodology": 0.12,
    "approach": 0.12,
    "framework": 0.14,
    "overview": 0.22,
    "inter-modal attention": 0.20,
    "intra-modal attention": 0.20,
    "joint learning": 0.18,
    "feature extraction": 0.12,
    "implementation details": 0.08,
}
LOW_VALUE_PAPER_SECTIONS = {
    "related work": 0.14,
    "ablation": 0.12,
    "experiments": 0.08,
    "comparisons": 0.08,
    "references": 0.60,
}
PAPER_METHOD_CONTENT_MARKERS = {
    "we propose": 0.10,
    "proposed semantic-guided representation enhancement": 0.14,
    "consists of an inter-modal attention module and an intra-modal attention module": 0.18,
    "inter-modal attention module": 0.12,
    "intra-modal attention module": 0.12,
    "joint learning mechanism": 0.12,
    "overall description": 0.08,
    "detailed pipeline": 0.08,
    "label semantics": 0.06,
}


def _where_filter(dataset_id: str, source_filter: SourceFilter):
    if source_filter == "both":
        return {"dataset_id": dataset_id}
    return {
        "$and": [
            {"dataset_id": dataset_id},
            {"source_type": source_filter},
        ]
    }


def _source_query(question: str, source_filter: SourceFilter) -> str:
    if source_filter == "repo":
        return f"{question} {REPO_QUERY_EXPANSION}"
    if source_filter == "paper" and _is_method_implementation_question(question):
        return f"{question} {PAPER_METHOD_QUERY_EXPANSION}"
    return question


def _repo_path_priority(file_path: str) -> float:
    normalized_path = file_path.replace("\\", "/").lower().strip("/")
    filename = normalized_path.rsplit("/", maxsplit=1)[-1]
    priority = IMPORTANT_REPO_FILES.get(normalized_path, 0.0)

    if normalized_path.startswith("models/") and normalized_path.endswith(".py"):
        priority += 0.04

    priority -= GENERIC_REPO_FILE_PENALTIES.get(filename, 0.0)
    for path_part, penalty in NOISY_REPO_PATH_PENALTIES.items():
        if path_part in normalized_path:
            priority -= penalty

    return priority


def _repo_content_priority(content: str) -> float:
    normalized_content = content.lower()
    return sum(
        boost
        for marker, boost in CORE_REPO_CONTENT_MARKERS.items()
        if marker in normalized_content
    )


def _is_method_implementation_question(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in IMPLEMENTATION_METHOD_QUERY_MARKERS)


def _is_low_value_repo_path(file_path: str) -> bool:
    normalized_path = file_path.replace("\\", "/").lower().strip("/")
    filename = normalized_path.rsplit("/", maxsplit=1)[-1]
    return (
        any(path_part in normalized_path for path_part in NOISY_REPO_PATH_PENALTIES)
        or filename in GENERIC_REPO_FILE_PENALTIES
    )


def _is_high_value_repo_path(file_path: str) -> bool:
    normalized_path = file_path.replace("\\", "/").lower().strip("/")
    return normalized_path in IMPORTANT_REPO_FILES


def _paper_section_priority(section_title: str) -> float:
    normalized_title = section_title.lower()
    priority = sum(
        boost for marker, boost in METHOD_SECTIONS.items() if marker in normalized_title
    )
    priority -= sum(
        penalty for marker, penalty in LOW_VALUE_PAPER_SECTIONS.items() if marker in normalized_title
    )
    return priority


def _paper_content_priority(content: str) -> float:
    normalized_content = content.lower()
    return sum(
        boost
        for marker, boost in PAPER_METHOD_CONTENT_MARKERS.items()
        if marker in normalized_content
    )


def _is_low_value_paper_section(section_title: str) -> bool:
    normalized_title = section_title.lower()
    return any(marker in normalized_title for marker in LOW_VALUE_PAPER_SECTIONS)


def _is_high_value_paper_section(section_title: str) -> bool:
    return _paper_section_priority(section_title) > 0


def _ranking_score(item: RetrievedChunk) -> float:
    score = item.score if item.score is not None else 0.0
    if item.source_type == "repo":
        return (
            score
            + _repo_path_priority(str(item.metadata.get("file_path", "")))
            + _repo_content_priority(item.content)
        )
    if item.source_type == "paper":
        return (
            score
            + _paper_section_priority(str(item.metadata.get("section_title", "")))
            + _paper_content_priority(item.content)
        )
    return score


def _augment_method_candidates(
    collection,
    dataset_id: str,
    source_filter: SourceFilter,
    retrieved: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    existing_chunk_ids = {str(item.metadata.get("chunk_id", "")) for item in retrieved}
    source_items = collection.get(
        where=_where_filter(dataset_id, source_filter),
        include=["documents", "metadatas"],
    )
    documents = source_items.get("documents", [])
    metadatas = source_items.get("metadatas", [])

    augmented: list[RetrievedChunk] = []
    for content, metadata in zip(documents, metadatas):
        chunk_id = str(metadata.get("chunk_id", ""))
        if not chunk_id or chunk_id in existing_chunk_ids:
            continue

        source_type = metadata.get("source_type")
        if source_type != source_filter:
            continue

        if source_filter == "repo":
            if not _is_high_value_repo_path(str(metadata.get("file_path", ""))):
                continue
        elif source_filter == "paper":
            section_title = str(metadata.get("section_title", ""))
            if not _is_high_value_paper_section(section_title):
                continue
            if _is_low_value_paper_section(section_title):
                continue

        augmented.append(
            RetrievedChunk(
                content=content,
                metadata=metadata,
                score=0.0,
                source_type=source_type,
            )
        )

    return augmented


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

    if source_filter == "both":
        combined_results = retrieve_evidence(question, dataset_id, source_filter="paper", top_k=top_k)
        combined_results.extend(retrieve_evidence(question, dataset_id, source_filter="repo", top_k=top_k))
        return sorted(combined_results, key=_ranking_score, reverse=True)

    if _is_method_implementation_question(question):
        result_count = max(1, min(max(top_k * 8, top_k), total_count))
    else:
        result_count = max(1, min(max(top_k * 4, top_k), total_count))
    results = collection.query(
        query_texts=[_source_query(question, source_filter)],
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

    if _is_method_implementation_question(question):
        retrieved.extend(_augment_method_candidates(collection, dataset_id, source_filter, retrieved))

    ranked = sorted(retrieved, key=_ranking_score, reverse=True)
    if source_filter == "repo" and _is_method_implementation_question(question):
        high_value_results = [
            item
            for item in ranked
            if not _is_low_value_repo_path(str(item.metadata.get("file_path", "")))
        ]
        if high_value_results:
            low_value_results = [item for item in ranked if item not in high_value_results]
            ranked = high_value_results + low_value_results

    if source_filter == "paper" and _is_method_implementation_question(question):
        high_value_results = [
            item
            for item in ranked
            if not _is_low_value_paper_section(str(item.metadata.get("section_title", "")))
        ]
        if high_value_results:
            low_value_results = [item for item in ranked if item not in high_value_results]
            ranked = high_value_results + low_value_results

    return ranked[:top_k]
