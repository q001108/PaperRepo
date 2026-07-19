from __future__ import annotations

import hashlib
import logging
import json
import os
from pathlib import Path
import re
from typing import Literal

from src.llm_client import chat_completion_content, is_llm_configured
from src.schemas import AgentAnswer, AnswerEvidence, RetrievedChunk, RouteDecision


logger = logging.getLogger(__name__)
MAX_EVIDENCE_ITEMS_PER_SOURCE = 3
MAX_SNIPPET_CHARS = 420
MAX_LLM_EVIDENCE_CHARS = 1200
MAX_SUMMARY_SENTENCES = 2
SUPPORTED_CONFIDENCE_THRESHOLD = 0.60
LLM_ANSWER_CACHE_VERSION = "answer-cache-v2"
METHOD_SENTENCE_MARKERS = {
    "propose",
    "proposed",
    "introduce",
    "introduced",
    "present",
    "presented",
    "framework",
    "method",
    "approach",
    "model",
    "module",
    "consist",
    "consists",
    "component",
    "components",
    "提出",
    "方法",
    "框架",
    "模型",
    "模块",
    "包含",
    "组成",
    "采用",
    "使用",
}
BACKGROUND_SENTENCE_MARKERS = {
    "drawback",
    "drawbacks",
    "however",
    "challenging",
    "problem",
    "problems",
    "limitation",
    "limitations",
    "不足",
    "然而",
    "挑战",
    "问题",
    "缺点",
}
STRONG_METHOD_MARKERS = {
    "in this paper, we propose",
    "we propose",
    "we introduce",
    "proposed framework consists",
    "framework consists",
    "本文提出",
    "该框架包含",
    "框架包含",
}
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
CITATION_PATTERN = re.compile(r"\[(?:P|R)\d+\]")


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


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _sentence_score(sentence: str) -> int:
    lowered = sentence.lower()
    score = sum(1 for marker in METHOD_SENTENCE_MARKERS if marker.lower() in lowered)
    score += 4 * sum(1 for marker in STRONG_METHOD_MARKERS if marker.lower() in lowered)
    if any(marker.lower() in lowered for marker in BACKGROUND_SENTENCE_MARKERS):
        score -= 3
    return score


def _method_summary_sentences(paper_evidence: list[AnswerEvidence]) -> list[str]:
    scored_sentences: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for evidence_index, item in enumerate(paper_evidence):
        for sentence in _split_sentences(item.content):
            normalized_sentence = sentence.lower()
            if normalized_sentence in seen:
                continue
            seen.add(normalized_sentence)
            score = _sentence_score(sentence)
            if score > 0:
                scored_sentences.append((score, evidence_index, sentence))

    if not scored_sentences and paper_evidence:
        return _split_sentences(paper_evidence[0].content)[:1]

    ranked_sentences = sorted(scored_sentences, key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    for _score, _evidence_index, sentence in ranked_sentences:
        if len(selected) >= MAX_SUMMARY_SENTENCES:
            break
        selected.append(_shorten(sentence))
    return selected


def _format_paper_evidence_list(paper_evidence: list[AnswerEvidence], language: Literal["en", "zh"]) -> str:
    lines = []
    for item in paper_evidence:
        page_start = item.metadata.get("page_num", "unknown")
        page_end = item.metadata.get("page_end")
        page_range = f"{page_start}-{page_end}" if page_end and page_end != page_start else str(page_start)
        section = item.metadata.get("section_title") or ("未知章节" if language == "zh" else "unknown section")
        if language == "zh":
            lines.append(f"- [{item.evidence_id}] 第 {page_range} 页，{section}：{_shorten(item.content)}")
        else:
            lines.append(f"- [{item.evidence_id}] page {page_range}, {section}: {_shorten(item.content)}")
    return "\n".join(lines)


def _llm_answer_config() -> tuple[str, str]:
    provider = (
        os.getenv("LLM_ANSWER_PROVIDER", "").strip().lower()
        or os.getenv("LLM_ROUTER_PROVIDER", "").strip().lower()
    )
    model = os.getenv("LLM_ANSWER_MODEL", "").strip() or os.getenv("LLM_ROUTER_MODEL", "").strip()
    return provider, model


def _answer_cache_enabled() -> bool:
    return os.getenv("LLM_ANSWER_CACHE_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _answer_cache_path() -> Path:
    return Path(os.getenv("LLM_ANSWER_CACHE_PATH", "").strip() or ".llm_cache/answers.json")


def _read_answer_cache() -> dict[str, dict[str, object]]:
    if not _answer_cache_enabled():
        return {}

    cache_path = _answer_cache_path()
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unable to read LLM answer cache %s: %s", cache_path, exc)
        return {}

    return data if isinstance(data, dict) else {}


def _write_answer_cache(cache: dict[str, dict[str, object]]) -> None:
    if not _answer_cache_enabled():
        return

    cache_path = _answer_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(cache_path)
    except Exception as exc:
        logger.warning("Unable to write LLM answer cache %s: %s", cache_path, exc)


def _metadata_for_cache(metadata: dict[str, str | int | float | bool]) -> dict[str, str | int | float | bool]:
    cache_fields = {
        "chunk_id",
        "dataset_id",
        "source_type",
        "source_file",
        "page_num",
        "page_end",
        "section_title",
        "file_path",
        "line_start",
        "line_end",
        "repo_url",
        "repo_commit_hash",
        "language",
    }
    return {key: value for key, value in sorted(metadata.items()) if key in cache_fields}


def _evidence_for_cache(item: AnswerEvidence) -> dict[str, object]:
    return {
        "evidence_id": item.evidence_id,
        "source_type": item.source_type,
        "metadata": _metadata_for_cache(item.metadata),
        "content_sha256": hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
    }


def _llm_answer_cache_key(
    provider: str,
    model: str,
    question: str,
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    language: Literal["en", "zh"],
) -> str:
    payload = {
        "version": LLM_ANSWER_CACHE_VERSION,
        "provider": provider,
        "model": model,
        "language": language,
        "question": " ".join(question.split()),
        "question_type": route_decision.question_type,
        "source_filter": route_decision.source_filter,
        "paper_evidence": [_evidence_for_cache(item) for item in paper_evidence],
        "repo_evidence": [_evidence_for_cache(item) for item in repo_evidence],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cached_llm_answer(cache_key: str) -> tuple[str, list[str]] | None:
    cache_entry = _read_answer_cache().get(cache_key)
    if not isinstance(cache_entry, dict):
        return None

    answer = str(cache_entry.get("answer", "")).strip()
    if not answer:
        return None

    limitations = cache_entry.get("limitations", [])
    if not isinstance(limitations, list):
        limitations = []
    return answer, [str(item) for item in limitations if str(item).strip()]


def _store_llm_answer(cache_key: str, answer: str, limitations: list[str]) -> None:
    cache = _read_answer_cache()
    cache[cache_key] = {
        "version": LLM_ANSWER_CACHE_VERSION,
        "answer": answer,
        "limitations": limitations,
    }
    _write_answer_cache(cache)


def _evidence_reference(item: AnswerEvidence, language: Literal["en", "zh"]) -> str:
    if item.source_type == "paper":
        page_start = item.metadata.get("page_num", "unknown")
        page_end = item.metadata.get("page_end")
        page_range = f"{page_start}-{page_end}" if page_end and page_end != page_start else str(page_start)
        section = item.metadata.get("section_title") or ("未知章节" if language == "zh" else "unknown section")
        return f"{item.evidence_id} | paper | page {page_range} | section {section}"

    file_path = item.metadata.get("file_path", "unknown file")
    line_start = item.metadata.get("line_start", "unknown")
    line_end = item.metadata.get("line_end", "unknown")
    return f"{item.evidence_id} | repo | {file_path}:{line_start}-{line_end}"


def _llm_evidence_context(
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    language: Literal["en", "zh"],
) -> str:
    lines: list[str] = []
    for item in paper_evidence + repo_evidence:
        content = " ".join(item.content.split())
        if len(content) > MAX_LLM_EVIDENCE_CHARS:
            content = content[: MAX_LLM_EVIDENCE_CHARS - 3] + "..."
        lines.append(f"[{_evidence_reference(item, language)}]\n{content}")
    return "\n\n".join(lines)


def _short_evidence_label(item: AnswerEvidence, language: Literal["en", "zh"]) -> str:
    if item.source_type == "paper":
        page_start = item.metadata.get("page_num", "unknown")
        page_end = item.metadata.get("page_end")
        page_range = f"{page_start}-{page_end}" if page_end and page_end != page_start else str(page_start)
        section = item.metadata.get("section_title") or ("未知章节" if language == "zh" else "unknown section")
        if language == "zh":
            return f"[{item.evidence_id}] 论文，第 {page_range} 页，{section}"
        return f"[{item.evidence_id}] paper, page {page_range}, {section}"

    file_path = item.metadata.get("file_path", "unknown file")
    line_start = item.metadata.get("line_start", "unknown")
    line_end = item.metadata.get("line_end", "unknown")
    if language == "zh":
        return f"[{item.evidence_id}] 仓库，{file_path}:{line_start}-{line_end}"
    return f"[{item.evidence_id}] repository, {file_path}:{line_start}-{line_end}"


def _ensure_visible_citations(
    answer: str,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    language: Literal["en", "zh"],
) -> str:
    if CITATION_PATTERN.search(answer):
        return answer

    evidence_items = paper_evidence[:2] + repo_evidence[:2]
    if len(evidence_items) < 4:
        selected_ids = {item.evidence_id for item in evidence_items}
        evidence_items.extend(
            item
            for item in paper_evidence + repo_evidence
            if item.evidence_id not in selected_ids
        )
    if not evidence_items:
        return answer

    citation_ids = "".join(f"[{item.evidence_id}]" for item in evidence_items[:4])
    evidence_lines = "\n".join(
        f"- {_short_evidence_label(item, language)}" for item in evidence_items[:4]
    )
    if language == "zh":
        return f"{answer}\n\n上述结论依据：{citation_ids}\n\n证据：\n{evidence_lines}"
    return f"{answer}\n\nThis conclusion is based on: {citation_ids}\n\nEvidence:\n{evidence_lines}"


def _answer_conclusion(answer: str) -> str:
    cleaned = answer.strip()
    for marker in (
        "\n\nSupporting evidence",
        "\n\n支撑证据",
        "\n\nEvidence:",
        "\n\n证据：",
        "\n\n证据:",
        "\n\n上述结论依据",
        "\n\nThis conclusion is based on",
    ):
        marker_index = cleaned.find(marker)
        if marker_index >= 0:
            cleaned = cleaned[:marker_index].strip()
    return cleaned


def _score_label(score: float | None) -> str:
    return f"{score:.3f}" if score is not None else "N/A"


def _standard_evidence_line(item: AnswerEvidence, language: Literal["en", "zh"]) -> str:
    score = _score_label(item.score)
    summary = _shorten(item.content)

    if item.source_type == "paper":
        page_start = item.metadata.get("page_num", "unknown")
        page_end = item.metadata.get("page_end")
        page_range = f"{page_start}-{page_end}" if page_end and page_end != page_start else str(page_start)
        section = item.metadata.get("section_title") or ("未知章节" if language == "zh" else "unknown section")
        if language == "zh":
            return (
                f"- [{item.evidence_id}] 来源=论文 | 页码={page_range} | 章节={section} | 分数={score}\n"
                f"  摘要：{summary}"
            )
        return (
            f"- [{item.evidence_id}] source=paper | page={page_range} | section={section} | score={score}\n"
            f"  summary: {summary}"
        )

    file_path = item.metadata.get("file_path", "unknown file")
    line_start = item.metadata.get("line_start", "unknown")
    line_end = item.metadata.get("line_end", "unknown")
    if language == "zh":
        return (
            f"- [{item.evidence_id}] 来源=仓库 | 文件={file_path} | 行号={line_start}-{line_end} | 分数={score}\n"
            f"  摘要：{summary}"
        )
    return (
        f"- [{item.evidence_id}] source=repository | file={file_path} | lines={line_start}-{line_end} | score={score}\n"
        f"  summary: {summary}"
    )


def _format_standard_answer(
    answer: str,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    language: Literal["en", "zh"],
) -> str:
    conclusion = _answer_conclusion(answer)
    evidence_items = paper_evidence + repo_evidence

    if language == "zh":
        evidence_text = (
            "\n".join(_standard_evidence_line(item, language) for item in evidence_items)
            if evidence_items
            else "- 无可用证据。"
        )
        return f"### 结论\n\n{conclusion}\n\n### 依据证据\n\n{evidence_text}"

    evidence_text = (
        "\n".join(_standard_evidence_line(item, language) for item in evidence_items)
        if evidence_items
        else "- No available evidence."
    )
    return f"### Conclusion\n\n{conclusion}\n\n### Evidence\n\n{evidence_text}"


def _dedupe_limited(items: list[str], limit: int = 3) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(item.split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        selected.append(normalized)
        if len(selected) >= limit:
            break
    return selected


def _follow_up_questions(
    question_type: Literal["paper_question", "repo_question", "cross_source_check"],
    status: Literal["supported", "partial", "conflict", "missing", "insufficient"],
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    language: Literal["en", "zh"],
) -> list[str]:
    if language == "zh":
        insufficient = [
            "当前缺少哪些证据？",
            "是否只查看论文证据？",
            "是否只查看仓库证据？",
        ]
        by_route = {
            "paper_question": [
                "这篇论文的方法由哪些模块组成？",
                "这篇论文的实验结果如何支持该方法？",
                "这个仓库是否实现了论文中的这些模块？",
            ],
            "repo_question": [
                "这个仓库的核心实现文件在哪里？",
                "这些代码对应论文中的哪些模块？",
                "如何运行训练或评估脚本？",
            ],
            "cross_source_check": [
                "仓库中哪些文件对应论文的核心方法？",
                "论文中的模块在代码中分别如何实现？",
                "仓库实现和论文描述有哪些不一致？",
            ],
        }
    else:
        insufficient = [
            "What evidence is currently missing?",
            "Can you answer using only paper evidence?",
            "Can you answer using only repository evidence?",
        ]
        by_route = {
            "paper_question": [
                "What modules make up the paper's method?",
                "What experimental results support this method?",
                "Does the repository implement these paper modules?",
            ],
            "repo_question": [
                "Where are the repository's core implementation files?",
                "Which paper modules do these code files correspond to?",
                "How do I run the training or evaluation scripts?",
            ],
            "cross_source_check": [
                "Which repository files correspond to the paper's core method?",
                "How is each paper module implemented in the code?",
                "Where do the repository implementation and paper description differ?",
            ],
        }

    if status in {"missing", "insufficient"}:
        return _dedupe_limited(insufficient)

    suggestions = list(by_route[question_type])
    if question_type == "paper_question" and repo_evidence:
        suggestions.insert(0, by_route["cross_source_check"][0])
    if question_type == "repo_question" and paper_evidence:
        suggestions.insert(0, by_route["cross_source_check"][1])
    if status == "conflict":
        suggestions.insert(0, by_route["cross_source_check"][2])
    return _dedupe_limited(suggestions)


def _build_agent_answer(
    *,
    answer: str,
    status: Literal["supported", "partial", "conflict", "missing", "insufficient"],
    confidence: float,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    limitations: list[str],
    language: Literal["en", "zh"],
    question_type: Literal["paper_question", "repo_question", "cross_source_check"],
) -> AgentAnswer:
    return AgentAnswer(
        answer=_format_standard_answer(answer, paper_evidence, repo_evidence, language),
        status=status,
        confidence=confidence,
        paper_evidence=paper_evidence,
        repo_evidence=repo_evidence,
        limitations=limitations,
        follow_up_questions=_follow_up_questions(
            question_type,
            status,
            paper_evidence,
            repo_evidence,
            language,
        ),
    )


def _llm_answer_messages(
    question: str,
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    fallback_answer: str,
    language: Literal["en", "zh"],
) -> list[dict[str, str]]:
    evidence_context = _llm_evidence_context(paper_evidence, repo_evidence, language)
    output_language = "Chinese" if language == "zh" else "English"
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence-bound research assistant. Answer only from the provided evidence. "
                "Do not use outside knowledge. Every concrete claim must cite one or more evidence IDs "
                "such as [P1], [P2], [R1]. If the evidence is insufficient, say so explicitly. "
                "Return only JSON with this exact shape: "
                '{"answer":"final answer with citations","limitations":["short limitation"]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Output language: {output_language}\n"
                f"Question type: {route_decision.question_type}\n"
                f"Evidence:\n{evidence_context}\n\n"
                f"Deterministic draft, also evidence-bound:\n{fallback_answer}\n\n"
                f"Question: {question}\n\n"
                "Write only the concise conclusion text. Keep citations visible when useful. "
                "Do not add separate Evidence, Supporting evidence, or Limitations sections."
            ),
        },
    ]


def _try_llm_answer(
    question: str,
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    fallback_answer: str,
    language: Literal["en", "zh"],
) -> tuple[str, list[str], bool]:
    provider, model = _llm_answer_config()
    if not is_llm_configured(provider, model):
        return fallback_answer, [], False

    cache_key = _llm_answer_cache_key(
        provider,
        model,
        question,
        route_decision,
        paper_evidence,
        repo_evidence,
        language,
    )
    cached_answer = _cached_llm_answer(cache_key)
    if cached_answer is not None:
        answer, limitations = cached_answer
        cache_limitation = (
            "已命中本地大模型回答缓存，未重复请求 DeepSeek。"
            if language == "zh"
            else "Reused the local LLM answer cache; DeepSeek was not called again."
        )
        return answer, limitations + [cache_limitation], False

    try:
        content = chat_completion_content(
            provider,
            model,
            _llm_answer_messages(
                question,
                route_decision,
                paper_evidence,
                repo_evidence,
                fallback_answer,
                language,
            ),
            response_format_json=True,
            max_tokens=900,
            timeout=45,
        )
        data = json.loads(content)
        answer = str(data.get("answer", "")).strip()
        if not answer:
            return fallback_answer, [], False
        answer = _ensure_visible_citations(answer, paper_evidence, repo_evidence, language)
        limitations = data.get("limitations", [])
        if not isinstance(limitations, list):
            limitations = []
        clean_limitations = [str(item) for item in limitations if str(item).strip()]
        cache_write_limitation = (
            "未命中本地大模型回答缓存，已请求 DeepSeek 并写入缓存。"
            if language == "zh"
            else "Local LLM answer cache missed; called DeepSeek and stored the answer."
        )
        clean_limitations.append(cache_write_limitation)
        _store_llm_answer(cache_key, answer, clean_limitations)
        return answer, clean_limitations, True
    except Exception as exc:
        logger.warning("LLM answer generation failed: %s", exc)
        fallback_limitation = (
            "DeepSeek 回答生成失败，已回退到基于证据的模板回答。"
            if language == "zh"
            else "LLM answer generation failed; fell back to the evidence-bound template answer."
        )
        return fallback_answer, [fallback_limitation], False


def _llm_success_limitations(language: Literal["en", "zh"]) -> list[str]:
    provider, model = _llm_answer_config()
    if language == "zh":
        return [f"已使用 {provider}:{model} 基于检索证据进行回答组织。"]
    return [f"Used {provider}:{model} to synthesize the answer from retrieved evidence."]


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
    language: Literal["en", "zh"],
) -> AgentAnswer:
    missing_text = ", ".join(_source_label(source, language) for source in missing_sources)
    has_partial_evidence = bool(paper_evidence or repo_evidence)
    if language == "zh":
        return _build_agent_answer(
            answer=f"证据不足，无法安全回答。缺少必要证据来源：{missing_text}。不会给出无证据支持的结论。",
            status="partial" if has_partial_evidence else "missing",
            confidence=min(0.59, _average_score(paper_evidence + repo_evidence)) if has_partial_evidence else 0.0,
            paper_evidence=paper_evidence,
            repo_evidence=repo_evidence,
            limitations=[
                "回答器只使用检索到的证据，不会超出证据进行推断。",
                f"已选择路由：{_question_type_label(route_decision.question_type, language)}。",
            ],
            language=language,
            question_type=route_decision.question_type,
        )

    return _build_agent_answer(
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
        language=language,
        question_type=route_decision.question_type,
    )


def _cross_source_insufficient_answer(
    route_decision: RouteDecision,
    paper_evidence: list[AnswerEvidence],
    repo_evidence: list[AnswerEvidence],
    missing_sources: list[str],
    language: Literal["en", "zh"],
) -> AgentAnswer:
    missing_text = " and ".join(f"{source} evidence" for source in missing_sources)
    if language == "zh":
        zh_missing_text = "和".join(f"{_source_label(source, language)}证据" for source in missing_sources)
        return _build_agent_answer(
            answer=f"证据不足，无法完成跨来源核查。缺少必要的{zh_missing_text}。不会给出论文与仓库一致性的结论。",
            status="insufficient",
            confidence=0.0,
            paper_evidence=paper_evidence,
            repo_evidence=repo_evidence,
            limitations=[
                "跨来源核查需要同时具备来自论文和仓库的有效证据。",
                f"已选择路由：{_question_type_label(route_decision.question_type, language)}。",
            ],
            language=language,
            question_type=route_decision.question_type,
        )

    return _build_agent_answer(
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
        language=language,
        question_type=route_decision.question_type,
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


def _source_label(source: str, language: Literal["en", "zh"]) -> str:
    if language == "zh":
        return {"paper": "论文", "repo": "仓库"}.get(source, source)
    return source


def _question_type_label(question_type: str, language: Literal["en", "zh"]) -> str:
    if language == "zh":
        return {
            "paper_question": "论文问题",
            "repo_question": "仓库问题",
            "cross_source_check": "跨来源核查",
        }.get(question_type, question_type)
    return question_type


def _base_limitations(language: Literal["en", "zh"]) -> list[str]:
    if language == "zh":
        return [
            "此回答仅受检索到的片段约束。",
            "若启用大模型综合，也必须只使用检索证据并保留证据引用。",
        ]
    return [
        "This answer is constrained to retrieved chunks only.",
        "If LLM synthesis is enabled, it must stay within retrieved evidence and keep evidence citations.",
    ]


def generate_constrained_answer(
    question: str,
    route_decision: RouteDecision,
    retrieved_chunks: list[RetrievedChunk],
    language: Literal["en", "zh"] = "en",
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
            language,
        )

    if missing_sources:
        return _not_supported_answer(route_decision, paper_evidence, repo_evidence, missing_sources, language)

    limitations = _base_limitations(language)

    if route_decision.question_type == "paper_question":
        confidence = min(0.95, route_decision.confidence * _average_score(paper_evidence))
        status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
        summary_sentences = _method_summary_sentences(paper_evidence)
        evidence_list = _format_paper_evidence_list(paper_evidence, language)
        if language == "zh":
            if summary_sentences:
                method_summary = "；".join(summary_sentences)
                answer = (
                    f"根据检索到的论文片段，这篇论文描述/提出的方法可以概括为：{method_summary}\n\n"
                    f"支撑证据：\n{evidence_list}"
                )
            else:
                answer = f"检索到的论文证据不足以形成明确方法概述。\n\n支撑证据：\n{evidence_list}"
        else:
            if summary_sentences:
                method_summary = " ".join(summary_sentences)
                answer = (
                    f"Based on the retrieved paper snippets, the paper's method can be summarized as: "
                    f"{method_summary}\n\n"
                    f"Supporting evidence:\n{evidence_list}"
                )
            else:
                answer = f"The retrieved paper evidence is not enough to form a clear method summary.\n\nSupporting evidence:\n{evidence_list}"
        answer, llm_limitations, llm_used = _try_llm_answer(
            question,
            route_decision,
            paper_evidence,
            [],
            answer,
            language,
        )
        return _build_agent_answer(
            answer=answer,
            status=status,
            confidence=confidence,
            paper_evidence=paper_evidence,
            repo_evidence=[],
            limitations=limitations
            + (_llm_success_limitations(language) if llm_used else [])
            + llm_limitations,
            language=language,
            question_type=route_decision.question_type,
        )

    if route_decision.question_type == "repo_question":
        best = repo_evidence[0]
        confidence = min(0.95, route_decision.confidence * _average_score(repo_evidence))
        status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
        file_path = best.metadata.get("file_path", "unknown file")
        line_start = best.metadata.get("line_start", "unknown")
        line_end = best.metadata.get("line_end", "unknown")
        if language == "zh":
            answer = (
                f"检索到的仓库证据可以支持一个有限的仓库侧回答。"
                f"关键仓库证据 [R1] 来自 {file_path}:{line_start}-{line_end}："
                f"{_shorten(best.content)}"
            )
        else:
            answer = (
                f"The retrieved repository evidence supports a limited repo-only answer. "
                f"Key repository evidence [R1] from {file_path}:{line_start}-{line_end}: "
                f"{_shorten(best.content)}"
            )
        answer, llm_limitations, llm_used = _try_llm_answer(
            question,
            route_decision,
            [],
            repo_evidence,
            answer,
            language,
        )
        return _build_agent_answer(
            answer=answer,
            status=status,
            confidence=confidence,
            paper_evidence=[],
            repo_evidence=repo_evidence,
            limitations=limitations
            + (_llm_success_limitations(language) if llm_used else [])
            + llm_limitations,
            language=language,
            question_type=route_decision.question_type,
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
        if language == "zh":
            answer = (
                "检索到的论文证据和仓库证据看起来存在冲突。"
                f"论文侧证据 [P1]：{_shorten(paper_best.content)} "
                f"仓库侧证据 [R1] 来自 {repo_path}:{repo_start}-{repo_end}：{_shorten(repo_best.content)}"
            )
        else:
            answer = (
                "The retrieved paper and repository evidence appear to conflict. "
                f"Paper-side evidence [P1]: {_shorten(paper_best.content)} "
                f"Repository-side evidence [R1] from {repo_path}:{repo_start}-{repo_end}: {_shorten(repo_best.content)}"
            )
        answer, llm_limitations, llm_used = _try_llm_answer(
            question,
            route_decision,
            paper_evidence,
            repo_evidence,
            answer,
            language,
        )
        return _build_agent_answer(
            answer=answer,
            status="conflict",
            confidence=max(confidence, SUPPORTED_CONFIDENCE_THRESHOLD),
            paper_evidence=paper_evidence,
            repo_evidence=repo_evidence,
            limitations=limitations
            + (_llm_success_limitations(language) if llm_used else [])
            + llm_limitations
            + (
                [
                    "冲突检测基于检索片段中的显式矛盾标记。",
                    "不会执行仓库代码。",
                ]
                if language == "zh"
                else [
                    "Conflict detection is based on explicit contradiction markers in retrieved snippets.",
                    "No repository code is executed.",
                ]
            ),
            language=language,
            question_type=route_decision.question_type,
        )

    status = "supported" if confidence >= SUPPORTED_CONFIDENCE_THRESHOLD else "partial"
    if language == "zh":
        answer = (
            "当前有跨来源证据，可进行有限的一致性核查。"
            f"论文侧发现 [P1]：{_shorten(paper_best.content)} "
            f"仓库侧发现 [R1] 来自 {repo_path}:{repo_start}-{repo_end}：{_shorten(repo_best.content)} "
            "结论仅限于这些被引用的片段 [P1][R1]。"
        )
    else:
        answer = (
            "Cross-source evidence is available for a limited consistency check. "
            f"Paper-side finding [P1]: {_shorten(paper_best.content)} "
            f"Repository-side finding [R1] from {repo_path}:{repo_start}-{repo_end}: {_shorten(repo_best.content)} "
            "The conclusion is limited to these cited snippets [P1][R1]."
        )
    answer, llm_limitations, llm_used = _try_llm_answer(
        question,
        route_decision,
        paper_evidence,
        repo_evidence,
        answer,
        language,
    )
    return _build_agent_answer(
        answer=answer,
        status=status,
        confidence=confidence,
        paper_evidence=paper_evidence,
        repo_evidence=repo_evidence,
        limitations=limitations
        + (_llm_success_limitations(language) if llm_used else [])
        + llm_limitations
        + (
            [
                "仅凭检索结果不能证明跨来源完全等价。",
                "未使用矛盾检测器，也不会执行代码。",
            ]
            if language == "zh"
            else [
                "Cross-source equivalence is not proven by retrieval alone.",
                "No contradiction detector or code execution is used.",
            ]
        ),
        language=language,
        question_type=route_decision.question_type,
    )
