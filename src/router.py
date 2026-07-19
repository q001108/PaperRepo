from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from src.llm_client import chat_completion_content, is_llm_configured
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
    "manuscript",
    "文章",
    "本文",
    "作者",
    "方法",
    "实验",
    "结果",
    "结论",
    "摘要",
    "章节",
    "提出",
    "采用",
    "使用",
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
    "脚本",
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
    "仓库是否实现",
    "代码是否实现",
    "匹配",
    "比较",
    "一致性",
}

PAPER_CONTEXT_KEYWORDS = {
    "paper",
    "article",
    "manuscript",
    "论文",
    "文章",
    "本文",
    "这篇论文",
}
REPO_CONTEXT_KEYWORDS = {
    "repo",
    "repository",
    "github",
    "code",
    "readme",
    "仓库",
    "代码",
}
PAPER_METHOD_QUESTIONS = {
    "论文实现了什么方法",
    "论文提出了什么方法",
    "论文用了什么方法",
    "论文使用了什么方法",
    "论文介绍了什么方法",
    "论文描述了什么方法",
    "这篇论文实现了什么方法",
    "这篇论文提出了什么方法",
    "这篇论文用了什么方法",
    "这篇论文使用了什么方法",
    "这篇论文介绍了什么方法",
    "这篇论文描述了什么方法",
    "本文实现了什么方法",
    "本文提出了什么方法",
    "本文用了什么方法",
    "本文使用了什么方法",
    "本文介绍了什么方法",
    "本文描述了什么方法",
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


def _contains_any(question: str, keywords: set[str]) -> bool:
    lowered = question.lower()
    for keyword in keywords:
        lowered_keyword = keyword.lower()
        if lowered_keyword.isascii() and " " not in lowered_keyword:
            if re.search(rf"\b{re.escape(lowered_keyword)}\b", lowered):
                return True
        elif lowered_keyword in lowered:
            return True
    return False


def _rule_route(question: str) -> RouteDecision | None:
    lowered_question = question.lower()
    paper_hits = _count_keyword_hits(question, PAPER_KEYWORDS)
    repo_hits = _count_keyword_hits(question, REPO_KEYWORDS)
    cross_hits = _count_keyword_hits(question, CROSS_SOURCE_KEYWORDS)
    has_paper_context = _contains_any(question, PAPER_CONTEXT_KEYWORDS)
    has_repo_context = _contains_any(question, REPO_CONTEXT_KEYWORDS)

    if any(pattern in lowered_question for pattern in PAPER_METHOD_QUESTIONS):
        return RouteDecision(
            question_type="paper_question",
            source_filter="paper",
            confidence=0.9,
            rationale="The question asks what method the paper proposes or uses.",
            method="rules",
        )

    if cross_hits > 0 or (has_paper_context and has_repo_context):
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

    if not is_llm_configured(provider, model):
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user question into exactly one type: "
                "paper_question, repo_question, or cross_source_check. "
                "Return only JSON with this exact shape: "
                '{"question_type":"paper_question","confidence":0.8,"rationale":"short reason"}.'
            ),
        },
        {"role": "user", "content": question},
    ]

    try:
        content = chat_completion_content(
            provider,
            model,
            messages,
            response_format_json=True,
            max_tokens=300,
            timeout=20,
        )
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
