from __future__ import annotations

import logging

from src.schemas import AgentAnswer, AnswerEvidence, RetrievedChunk, RouteDecision


logger = logging.getLogger(__name__)
MAX_EVIDENCE_ITEMS_PER_SOURCE = 3
MAX_SNIPPET_CHARS = 420
SUPPORTED_CONFIDENCE_THRESHOLD = 0.60
CONFLICT_MARKERS = {
    "does not",
    "do not",
    "not implement",
    "not implemented",
    "not use",
    "without",
    "unsupported",
    "no implementation",
    "不",
    "未",
    "没有",
    "不一致",
}
SUPPORT_MARKERS = {
    "implement",
    "implemented",
    "implements",
    "use",
    "uses",
    "support",
    "supports",
    "propose",
    "proposes",
    "实现",
    "使用",
    "支持",
    "提出",
}


def _make_evidence(
    prefix: str,
    chunks: list[RetrievedChunk],
) -> list[AnswerEvidence]:
    evidence_items: list[AnswerEvidence] = []
    for index, chunk in enumerate(chunks[:MAX_EVIDENCE_ITEMS_PER_SOURCE], start=1):
        evidence_items.append(
            AnswerEvidence(
                evidence_id=f"{prefix}{index}",
                source_type=chunk.source_type,
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
            )
        )
    return evidence_items


def _shorten(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= MAX_SNIPPET_CHARS:
        return normalized
    return normalized[: MAX_SNIPPET_CHARS - 3] + "..."


def _average_score(items: list[AnswerEvidence]) -> float:
    scores = [item.score for item in items if item.score is not None]
    if not scores:
        return 0.35
    return sum(scores) / len(scores)


def _not_supported_answer(
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    missing_sources: list[str],
) -> AgentAnswer:
    missing_text = ", ".join(missing_sources)
    has_partial_evidence = bool(paper_evidence or repo_evidence)
    return AgentAnswer(
        answer=(
            f"Insufficient evidence to answer safely. Missing required evidence from: {missing_text}. "
            "No unsupported conclusion is provided."
        ),
        status="partial" if has_partial_evidence else "missing",
        confidence=min(0.59, _average_score(paper_evidence + repo_evidence)) if has_partial_evidence else 0.0,
        paper_evidence=paper_evidence,
        repo_evidence=repo_evidence,
        limitations=[
            "The answerer only uses retrieved evidence and will not infer beyond it.",
            f"Route selected: {route_decision.question_type}.",
        ],
    )


def _cross_source_insufficient_answer(
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    missing_sources: list[str],
) -> AgentAnswer:
    missing_text = " and ".join(f"{source} evidence" for source in missing_sources)
    return AgentAnswer(
        answer=(
            f"Insufficient evidence to complete the cross-source check. Missing required {missing_text}. "
            "No conclusion about paper-repository consistency is provided."
        ),
        status="insufficient",
        confidence=0.0,
        paper_evidence=paper_evidence,
        repo_evidence=repo_evidence,
        limitations=[
            "Cross-source checks require valid evidence from both the paper and the repository.",
            f"Route selected: {route_decision.question_type}.",
        ],
    )


def _has_marker(text: str, markers: set[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _has_conflict(paper_evidence: list[AnswerEvidence], repo_evidence: list[AnswerEvidence]) -> bool:
    paper_text = " ".join(item.content for item in paper_evidence)
    repo_text = " ".join(item.content for item in repo_evidence)

    paper_negative = _has_marker(paper_text, CONFLICT_MARKERS)
    repo_negative = _has_marker(repo_text, CONFLICT_MARKERS)
    paper_positive = _has_marker(paper_text, SUPPORT_MARKERS)
    repo_positive = _has_marker(repo_text, SUPPORT_MARKERS)

    return (paper_positive and repo_negative) or (repo_positive and paper_negative)


def generate_constrained_answer(
    question: str,
    route_decision: RouteDecision,
    retrieved_chunks: list[RetrievedChunk],
) -> AgentAnswer:
    """Generate a conservative answer where each claim cites retrieved evidence."""
    logger.info("Generating constrained answer for route %s", route_decision.question_type)

    paper_chunks = [chunk for chunk in retrieved_chunks if chunk.source_type == "paper"]
    repo_chunks = [chunk for chunk in retrieved_chunks if chunk.source_type == "repo"]
    paper_evidence = _make_evidence("P", paper_chunks)
    repo_evidence = _make_evidence("R", repo_chunks)

    required_sources = {
        "paper_question": ["paper"],
        "repo_question": ["repo"],
        "cross_source_check": ["paper", "repo"],
    }[route_decision.question_type]
    missing_sources = []
    if "paper" in required_sources and not paper_evidence:
        missing_sources.append("paper")
    if "repo" in required_sources and not repo_evidence:
        missing_sources.append("repo")

    if missing_sources and route_decision.question_type == "cross_source_check":
        return _cross_source_insufficient_answer(
            route_decision,
            paper_evidence,
            repo_evidence,
            missing_sources,
        )

    if missing_sources:
        return _not_supported_answer(route_decision, paper_evidence, repo_evidence, missing_sources)

    limitations = [
        "This answer is constrained to retrieved chunks only.",
        "It does not call an LLM for synthesis beyond evidence-bound templating.",
    ]

    if route_decision.question_type == "paper_question":
        best = paper_evidence[0]
        confidence = min(0.95, route_decision.confidence * _average_score(paper_evidence))
        status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
        answer = (
            f"The retrieved paper evidence supports a limited paper-only answer. "
            f"Key paper evidence [P1]: {_shorten(best.content)}"
        )
        return AgentAnswer(
            answer=answer,
            status=status,
            confidence=confidence,
            paper_evidence=paper_evidence,
            repo_evidence=[],
            limitations=limitations,
        )

    if route_decision.question_type == "repo_question":
        best = repo_evidence[0]
        confidence = min(0.95, route_decision.confidence * _average_score(repo_evidence))
        status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
        file_path = best.metadata.get("file_path", "unknown file")
        line_start = best.metadata.get("line_start", "unknown")
        line_end = best.metadata.get("line_end", "unknown")
        answer = (
            f"The retrieved repository evidence supports a limited repo-only answer. "
            f"Key repository evidence [R1] from {file_path}:{line_start}-{line_end}: "
            f"{_shorten(best.content)}"
        )
        return AgentAnswer(
            answer=answer,
            status=status,
            confidence=confidence,
            paper_evidence=[],
            repo_evidence=repo_evidence,
            limitations=limitations,
        )

    paper_best = paper_evidence[0]
    repo_best = repo_evidence[0]
    repo_path = repo_best.metadata.get("file_path", "unknown file")
    repo_start = repo_best.metadata.get("line_start", "unknown")
    repo_end = repo_best.metadata.get("line_end", "unknown")
    confidence = min(
        0.9,
        route_decision.confidence * _average_score([paper_best, repo_best]),
    )

    if _has_conflict(paper_evidence, repo_evidence):
        answer = (
            "The retrieved paper and repository evidence appear to conflict. "
            f"Paper-side evidence [P1]: {_shorten(paper_best.content)} "
            f"Repository-side evidence [R1] from {repo_path}:{repo_start}-{repo_end}: {_shorten(repo_best.content)}"
        )
        return AgentAnswer(
            answer=answer,
            status="conflict",
            confidence=max(confidence, SUPPORTED_CONFIDENCE_THRESHOLD),
            paper_evidence=paper_evidence,
            repo_evidence=repo_evidence,
            limitations=limitations
            + [
                "Conflict detection is based on explicit contradiction markers in retrieved snippets.",
                "No repository code is executed.",
            ],
        )

    status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
    answer = (
        "Cross-source evidence is available for a limited consistency check. "
        f"Paper-side finding [P1]: {_shorten(paper_best.content)} "
        f"Repository-side finding [R1] from {repo_path}:{repo_start}-{repo_end}: {_shorten(repo_best.content)} "
        "The conclusion is limited to these cited snippets [P1][R1]."
    )
    return AgentAnswer(
        answer=answer,
        status=status,
        confidence=confidence,
        paper_evidence=paper_evidence,
        repo_evidence=repo_evidence,
        limitations=limitations
        + [
            "Cross-source equivalence is not proven by retrieval alone.",
            "No contradiction detector or code execution is used.",
        ],
    )
