from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

from src.schemas import RouteDecision


logger = logging.getLogger(__name__)

PAPER_KEYWORDS = {
    "paper",
    "论文",
    "article",
    "abstract",
    "method",
    "methods",
    "experiment",
    "experiments",
    "result",
    "results",
    "section",
    "page",
    "作者",
    "方法",
    "实验",
    "结论",
    "摘要",
}
REPO_KEYWORDS = {
    "repo",
    "repository",
    "github",
    "code",
    "implementation",
    "implement",
    "readme",
    "requirements",
    "dockerfile",
    "function",
    "class",
    "script",
    "file",
    "代码",
    "仓库",
    "实现",
    "依赖",
    "函数",
    "文件",
}
CROSS_SOURCE_KEYWORDS = {
    "match",
    "matches",
    "compare",
    "consistent",
    "consistency",
    "reproduce",
    "reproduction",
    "paper and repo",
    "repo and paper",
    "connect",
    "alignment",
    "对应",
    "一致",
    "核查",
    "对比",
    "复现",
    "是否实现",
    "是否对应",
    "同时",
}


def _count_keyword_hits(question: str, keywords: set[str]) -> int:
    lowered = question.lower()
    hits = 0
    for keyword in keywords:
        lowered_keyword = keyword.lower()
        if lowered_keyword.isascii() and " " not in lowered_keyword:
            if re.search(rf"\b{re.escape(lowered_keyword)}\b", lowered):
                hits += 1
        elif lowered_keyword in lowered:
            hits += 1
    return hits


def _rule_route(question: str) -> RouteDecision | None:
    paper_hits = _count_keyword_hits(question, PAPER_KEYWORDS)
    repo_hits = _count_keyword_hits(question, REPO_KEYWORDS)
    cross_hits = _count_keyword_hits(question, CROSS_SOURCE_KEYWORDS)

    if cross_hits > 0 or (paper_hits > 0 and repo_hits > 0):
        return RouteDecision(
            question_type="cross_source_check",
            source_filter="both",
            confidence=0.85,
            rationale="The question asks to compare or connect paper evidence with repository evidence.",
            method="rules",
        )

    if repo_hits > paper_hits and repo_hits > 0:
        return RouteDecision(
            question_type="repo_question",
            source_filter="repo",
            confidence=0.8,
            rationale="The question mainly refers to repository files, code, dependencies, or implementation.",
            method="rules",
        )

    if paper_hits > 0:
        return RouteDecision(
            question_type="paper_question",
            source_filter="paper",
            confidence=0.8,
            rationale="The question mainly refers to paper sections, methods, experiments, or results.",
            method="rules",
        )

    return None


def _llm_route(question: str) -> RouteDecision | None:
    provider = os.getenv("LLM_ROUTER_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_ROUTER_MODEL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider != "openai" or not model or not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the user question into exactly one type: "
                    "paper_question, repo_question, or cross_source_check. "
                    "Return only JSON with question_type, confidence, and rationale."
                ),
            },
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data: dict[str, Any] = json.loads(content)
        question_type = data.get("question_type")
        source_filter = {
            "paper_question": "paper",
            "repo_question": "repo",
            "cross_source_check": "both",
        }.get(question_type)
        if source_filter is None:
            return None
        return RouteDecision(
            question_type=question_type,
            source_filter=source_filter,
            confidence=float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "LLM fallback route.")),
            method="llm",
        )
    except Exception as exc:
        logger.warning("LLM routing fallback failed: %s", exc)
        return None


def route_question(question: str) -> RouteDecision:
    """Route a question with rules first and optional LLM fallback."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    rule_decision = _rule_route(question)
    if rule_decision is not None:
        return rule_decision

    llm_decision = _llm_route(question)
    if llm_decision is not None:
        return llm_decision

    return RouteDecision(
        question_type="cross_source_check",
        source_filter="both",
        confidence=0.35,
        rationale="No clear routing keyword was found and LLM routing is not configured, so both sources are searched conservatively.",
        method="default",
    )
