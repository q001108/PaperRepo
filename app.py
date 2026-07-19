from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from src.answerer import generate_constrained_answer
from src.dataset import build_dataset_id, hash_file
from src.evidence_validator import filter_valid_evidence
from src.indexer import build_index, clear_dataset_index
from src.pdf_parser import assess_pdf_text_quality, parse_pdf
from src.retriever import retrieve_evidence
from src.repo_scanner import is_key_file_path, scan_repository_with_metadata, validate_github_url
from src.router import route_question
from src.schemas import AgentAnswer, AnswerEvidence, RetrievedChunk


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
MAX_CONVERSATION_TURNS = 4
MAX_CONTEXT_ANSWER_CHARS = 500

LANGUAGE_OPTIONS = {
    "English": "en",
    "中文": "zh",
}

UI_TEXT = {
    "en": {
        "language_label": "Language / 语言",
        "title": "PaperRepo Evidence-RAG Agent",
        "caption": "Upload a paper, add a public GitHub repository, and ask an evidence-bound question.",
        "current_dataset_caption": "Current dataset",
        "clear_index": "Clear current index",
        "clear_success": "Cleared index for dataset",
        "clear_error": "Failed to clear the current index. Check the logs for details.",
        "clear_conversation": "Clear conversation context",
        "clear_conversation_success": "Cleared conversation context.",
        "conversation_context": "Conversation context",
        "contextual_question": "Contextualized question",
        "paper_pdf": "Paper PDF",
        "github_url": "Public GitHub repository URL",
        "question": "Question",
        "question_placeholder": "Does the repository implement the method described in the paper?",
        "top_k": "Top-K evidence per source",
        "submit": "Rebuild Index and Run Agent",
        "idle_info": "The Agent will route the question, retrieve evidence, and answer only when evidence is available.",
        "missing_pdf": "Please upload a PDF file.",
        "missing_repo": "Please enter a GitHub repository URL.",
        "missing_question": "Please enter a question.",
        "unexpected_error": "Something went wrong while preparing the request. Check the logs for details.",
        "run_success": "Inputs accepted. Agent routing, retrieval, and constrained answering completed.",
        "parsed_pages": "Parsed PDF pages",
        "scanned_files": "Scanned files",
        "index_chunks": "Index chunks",
        "key_files_metric": "Key files",
        "dataset": "Dataset",
        "valid_evidence_metric": "Valid evidence",
        "text_quality": "PDF text quality",
        "extracted_pages": "Extracted pages",
        "low_text_pages": "Low-text pages",
        "readable_ratio": "Readable ratio",
        "ocr_recommended": "OCR recommended",
        "ocr_not_needed": "OCR not needed",
        "quality_ocr_detail": "Text extraction looks sparse or noisy; OCR fallback may be needed.",
        "quality_good_detail": "Text extraction looks usable; OCR is probably not needed.",
        "chunking_strategy": "Chunking strategy",
        "current_dataset": "Current dataset",
        "repository_url": "Repository URL",
        "commit_hash": "Commit hash",
        "pdf_hash": "PDF hash",
        "embedding_provider": "Embedding provider",
        "embedding_model": "Embedding model",
        "answer_llm": "Answer LLM",
        "unknown": "unknown",
        "key_repository_files": "Key repository files",
        "no_key_files": "No README, requirements.txt, environment.yml, or Dockerfile was found in the scanned files.",
        "agent_route": "Agent route",
        "question_type": "Question type",
        "search_source": "Search source",
        "route_confidence": "Route confidence",
        "routing_method": "Routing method",
        "agent_answer": "Agent answer",
        "answered_question": "Answered question",
        "status": "Status",
        "confidence": "Confidence",
        "bound_evidence": "Bound evidence",
        "follow_up_questions": "Suggested follow-up questions",
        "paper_evidence": "Paper evidence",
        "repository_evidence": "Repository evidence",
        "no_bound_evidence": "No evidence was strong enough for a supported answer.",
        "limitations": "Limitations",
        "paper_snippet": "Paper snippet",
        "section": "Section",
        "unknown_section": "Unknown section",
        "page": "page",
        "score": "score",
        "repo": "repo",
        "lines": "lines",
        "top_valid_evidence": "Top-K valid retrieved evidence per source",
        "no_route_evidence": "No evidence matched the selected route.",
        "raw_results": "Raw retrieval results",
        "filtered_results": "Filtered valid retrieval results",
        "structured_answer": "Structured agent answer",
    },
    "zh": {
        "language_label": "Language / 语言",
        "title": "PaperRepo 证据 RAG 智能体",
        "caption": "上传论文 PDF，填写公开 GitHub 仓库，并提出一个基于证据的问题。",
        "current_dataset_caption": "当前数据集",
        "clear_index": "清空当前索引",
        "clear_success": "已清空数据集索引",
        "clear_error": "清空当前索引失败，请查看日志了解详情。",
        "clear_conversation": "清空对话上下文",
        "clear_conversation_success": "已清空对话上下文。",
        "conversation_context": "对话上下文",
        "contextual_question": "上下文化后的问题",
        "paper_pdf": "论文 PDF",
        "github_url": "公开 GitHub 仓库 URL",
        "question": "问题",
        "question_placeholder": "这个仓库是否实现了论文中描述的方法？",
        "top_k": "每个来源的 Top-K 证据数量",
        "submit": "重建索引并运行智能体",
        "idle_info": "智能体会先判断问题类型，再检索证据，并且只在有证据支持时回答。",
        "missing_pdf": "请上传 PDF 文件。",
        "missing_repo": "请输入 GitHub 仓库 URL。",
        "missing_question": "请输入问题。",
        "unexpected_error": "准备请求时出错，请查看日志了解详情。",
        "run_success": "输入已接受，智能体已完成路由、检索和基于证据的回答。",
        "parsed_pages": "解析 PDF 页数",
        "scanned_files": "扫描文件数",
        "index_chunks": "索引片段数",
        "key_files_metric": "关键文件数",
        "dataset": "数据集",
        "valid_evidence_metric": "有效证据数",
        "text_quality": "PDF 文本质量",
        "extracted_pages": "可抽取文本页数",
        "low_text_pages": "低文本页数",
        "readable_ratio": "可读字符比例",
        "ocr_recommended": "建议后续使用 OCR",
        "ocr_not_needed": "暂不需要 OCR",
        "quality_ocr_detail": "PDF 可抽取文本偏少或噪声较多，后续可能需要 OCR 兜底。",
        "quality_good_detail": "PDF 文本抽取质量可用，暂时不需要 OCR。",
        "chunking_strategy": "分片策略",
        "current_dataset": "当前数据集",
        "repository_url": "仓库 URL",
        "commit_hash": "提交哈希",
        "pdf_hash": "PDF 哈希",
        "embedding_provider": "嵌入提供方",
        "embedding_model": "嵌入模型",
        "answer_llm": "回答大模型",
        "unknown": "未知",
        "key_repository_files": "关键仓库文件",
        "no_key_files": "扫描到的文件中没有 README、requirements.txt、environment.yml 或 Dockerfile。",
        "agent_route": "智能体路由",
        "question_type": "问题类型",
        "search_source": "检索来源",
        "route_confidence": "路由置信度",
        "routing_method": "路由方式",
        "agent_answer": "智能体回答",
        "answered_question": "本次回答对应的问题",
        "status": "状态",
        "confidence": "置信度",
        "bound_evidence": "绑定证据",
        "follow_up_questions": "推荐追问",
        "paper_evidence": "论文证据",
        "repository_evidence": "仓库证据",
        "no_bound_evidence": "没有足够强的证据支撑回答。",
        "limitations": "局限性",
        "paper_snippet": "论文片段",
        "section": "章节",
        "unknown_section": "未知章节",
        "page": "页",
        "score": "分数",
        "repo": "仓库",
        "lines": "行",
        "top_valid_evidence": "每个来源的 Top-K 有效检索证据",
        "no_route_evidence": "没有证据匹配当前路由。",
        "raw_results": "原始检索结果",
        "filtered_results": "过滤后的有效检索结果",
        "structured_answer": "结构化智能体回答",
    },
}

STATUS_LABELS = {
    "en": {
        "supported": "supported",
        "partial": "partial",
        "conflict": "conflict",
        "missing": "missing",
        "insufficient": "insufficient",
    },
    "zh": {
        "supported": "已支持",
        "partial": "部分支持",
        "conflict": "存在冲突",
        "missing": "缺少证据",
        "insufficient": "证据不足",
    },
}

ROUTE_LABELS = {
    "en": {
        "paper_question": "paper_question",
        "repo_question": "repo_question",
        "cross_source_check": "cross_source_check",
        "paper": "paper",
        "repo": "repo",
        "both": "both",
        "rules": "rules",
        "llm": "llm",
        "default": "default",
    },
    "zh": {
        "paper_question": "论文问题",
        "repo_question": "仓库问题",
        "cross_source_check": "跨来源核查",
        "paper": "论文",
        "repo": "仓库",
        "both": "论文和仓库",
        "rules": "规则",
        "llm": "大模型",
        "default": "默认",
    },
}

ROUTE_RATIONALES_ZH = {
    "paper_question": "问题主要指向论文的章节、方法、实验或结果。",
    "repo_question": "问题主要指向仓库文件、代码、依赖或实现细节。",
    "cross_source_check": "问题需要比较或连接论文证据与仓库证据。",
}


def _t(language: str, key: str) -> str:
    return UI_TEXT[language][key]


def _route_label(language: str, value: str) -> str:
    return ROUTE_LABELS[language].get(value, value)


def _status_label(language: str, value: str) -> str:
    return STATUS_LABELS[language].get(value, value)


def _page_range(metadata: dict[str, str | int | float | bool]) -> str:
    page_start = metadata.get("page_num", "N/A")
    page_end = metadata.get("page_end")
    if page_end and page_end != page_start:
        return f"{page_start}-{page_end}"
    return str(page_start)


def _route_rationale(language: str, question_type: str, rationale: str) -> str:
    if language == "zh":
        return ROUTE_RATIONALES_ZH.get(question_type, "未发现明确路由关键词，因此保守地检索两个来源。")
    return rationale


def _repo_code_expander_label(
    language: str,
    evidence_id: str | int,
    file_path: object,
    line_start: object,
    line_end: object,
    score: str,
) -> str:
    if language == "zh":
        return f"{evidence_id}. 查看代码证据 | {file_path}:{line_start}-{line_end} | 分数 {score}"
    return f"{evidence_id}. View code evidence | {file_path}:{line_start}-{line_end} | score {score}"


def _set_question_input(question: str) -> None:
    st.session_state["question_input"] = question


def _render_metric_grid(items: list[tuple[str, object]]) -> None:
    for offset in range(0, len(items), 3):
        row_items = items[offset : offset + 3]
        columns = st.columns(len(row_items))
        for column, (label, value) in zip(columns, row_items):
            column.metric(label, value)


def _source_requirements(source_filter: str) -> tuple[bool, bool]:
    return source_filter in {"paper", "both"}, source_filter in {"repo", "both"}


def _trim_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _answer_context_summary(answer: AgentAnswer) -> str:
    summary = answer.answer
    for marker in ("### Evidence", "### 依据证据"):
        marker_index = summary.find(marker)
        if marker_index >= 0:
            summary = summary[:marker_index]
    summary = summary.replace("### Conclusion", "").replace("### 结论", "")
    return _trim_text(summary, MAX_CONTEXT_ANSWER_CHARS)


def _build_contextual_question(question: str, history: list[dict[str, str]]) -> str:
    current_question = _trim_text(question, 600)
    recent_turns = [
        turn
        for turn in history[-MAX_CONVERSATION_TURNS:]
        if turn.get("question") or turn.get("answer")
    ]
    if not recent_turns:
        return current_question

    lines = [
        "Use the recent conversation context to resolve pronouns and follow-up references.",
        f"Current question: {current_question}",
        "Recent conversation:",
    ]
    for index, turn in enumerate(recent_turns, start=1):
        lines.append(f"{index}. User question: {_trim_text(turn.get('question', ''), 240)}")
        answer_summary = _trim_text(turn.get("answer", ""), MAX_CONTEXT_ANSWER_CHARS)
        if answer_summary:
            lines.append(f"   Answer summary: {answer_summary}")
        question_type = turn.get("question_type")
        source_filter = turn.get("source_filter")
        if question_type or source_filter:
            lines.append(f"   Route: {question_type or 'unknown'} / {source_filter or 'unknown'}")
    return "\n".join(lines)


def _append_conversation_turn(
    history: list[dict[str, str]],
    question: str,
    contextual_question: str,
    answer: AgentAnswer,
    question_type: str,
    source_filter: str,
) -> list[dict[str, str]]:
    updated_history = [
        *history,
        {
            "question": _trim_text(question, 600),
            "contextual_question": _trim_text(contextual_question, 1200),
            "answer": _answer_context_summary(answer),
            "status": answer.status,
            "question_type": question_type,
            "source_filter": source_filter,
        },
    ]
    return updated_history[-MAX_CONVERSATION_TURNS:]


def _save_uploaded_pdf(uploaded_file) -> Path:
    upload_dir = Path(".uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = upload_dir / safe_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _render_retrieved_chunk(index: int, item: RetrievedChunk, language: str) -> None:
    metadata = item.metadata
    score = f"{item.score:.3f}" if item.score is not None else "N/A"

    if item.source_type == "paper":
        page_range = _page_range(metadata)
        section_title = metadata.get("section_title") or _t(language, "unknown_section")
        st.markdown(
            f"**{index}. {_t(language, 'paper_evidence')}** | "
            f"{_t(language, 'page')} `{page_range}` | {_t(language, 'score')} `{score}`"
        )
        st.caption(f"{_t(language, 'section')}: {section_title}")
        st.text_area(
            _t(language, "paper_snippet"),
            item.content,
            height=160,
            key=f"paper-evidence-{index}",
            disabled=True,
        )
        return

    file_path = metadata.get("file_path", "N/A")
    line_start = metadata.get("line_start", "N/A")
    line_end = metadata.get("line_end", "N/A")
    with st.expander(
        _repo_code_expander_label(language, index, file_path, line_start, line_end, score),
        expanded=False,
    ):
        st.code(item.content, language=str(metadata.get("language") or "text").lower())


def _render_answer_evidence(item: AnswerEvidence, language: str) -> None:
    metadata = item.metadata
    score = f"{item.score:.3f}" if item.score is not None else "N/A"

    if item.source_type == "paper":
        page_range = _page_range(metadata)
        section_title = metadata.get("section_title") or _t(language, "unknown_section")
        st.markdown(
            f"**{item.evidence_id}** | {_route_label(language, 'paper')} "
            f"{_t(language, 'page')} `{page_range}` | {_t(language, 'score')} `{score}`"
        )
        st.caption(f"{_t(language, 'section')}: {section_title}")
        st.text_area(
            f"{_t(language, 'paper_evidence')} {item.evidence_id}",
            item.content,
            height=150,
            key=f"answer-paper-{item.evidence_id}",
            disabled=True,
        )
        return

    file_path = metadata.get("file_path", "N/A")
    line_start = metadata.get("line_start", "N/A")
    line_end = metadata.get("line_end", "N/A")
    with st.expander(
        _repo_code_expander_label(language, item.evidence_id, file_path, line_start, line_end, score),
        expanded=False,
    ):
        st.code(item.content, language=str(metadata.get("language") or "text").lower())


def _render_agent_answer(answer: AgentAnswer, language: str) -> None:
    st.subheader(_t(language, "agent_answer"))
    col1, col2 = st.columns(2)
    col1.metric(_t(language, "status"), _status_label(language, answer.status))
    col2.metric(_t(language, "confidence"), f"{answer.confidence:.2f}")
    st.markdown(answer.answer)

    if answer.follow_up_questions:
        st.subheader(_t(language, "follow_up_questions"))
        for index, follow_up_question in enumerate(answer.follow_up_questions, start=1):
            st.button(
                follow_up_question,
                key=f"follow-up-question-{index}",
                on_click=_set_question_input,
                args=(follow_up_question,),
                use_container_width=True,
            )

    st.subheader(_t(language, "bound_evidence"))
    if answer.paper_evidence:
        st.markdown(f"**{_t(language, 'paper_evidence')}**")
        for item in answer.paper_evidence:
            _render_answer_evidence(item, language)
    if answer.repo_evidence:
        st.markdown(f"**{_t(language, 'repository_evidence')}**")
        for item in answer.repo_evidence:
            _render_answer_evidence(item, language)
    if not answer.paper_evidence and not answer.repo_evidence:
        st.warning(_t(language, "no_bound_evidence"))

    st.subheader(_t(language, "limitations"))
    for limitation in answer.limitations:
        st.write(f"- {limitation}")


def main() -> None:
    st.set_page_config(
        page_title="PaperRepo Evidence-RAG Agent",
        page_icon="PR",
        layout="centered",
    )

    language_name = st.radio(
        UI_TEXT["en"]["language_label"],
        list(LANGUAGE_OPTIONS),
        index=0,
        horizontal=True,
    )
    language = LANGUAGE_OPTIONS[language_name]

    st.title(_t(language, "title"))
    st.caption(_t(language, "caption"))

    last_dataset_id = st.session_state.get("last_dataset_id")
    if last_dataset_id:
        st.caption(f"{_t(language, 'current_dataset_caption')}: `{last_dataset_id[:8]}`")
        if st.button(_t(language, "clear_index")):
            try:
                clear_dataset_index(last_dataset_id)
                st.success(f"{_t(language, 'clear_success')} `{last_dataset_id[:8]}`.")
            except Exception:
                logger.exception("Failed to clear current index")
                st.error(_t(language, "clear_error"))

    conversation_history = st.session_state.get("conversation_history", [])
    if not isinstance(conversation_history, list):
        conversation_history = []
        st.session_state["conversation_history"] = conversation_history
    if conversation_history:
        st.caption(f"{_t(language, 'conversation_context')}: {len(conversation_history)}")
        if st.button(_t(language, "clear_conversation")):
            st.session_state["conversation_history"] = []
            st.success(_t(language, "clear_conversation_success"))
            conversation_history = []

    uploaded_pdf = st.file_uploader(_t(language, "paper_pdf"), type=["pdf"])
    github_url = st.text_input(
        _t(language, "github_url"),
        placeholder="https://github.com/owner/repository",
    )
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    question = st.text_area(
        _t(language, "question"),
        placeholder=_t(language, "question_placeholder"),
        height=120,
        key="question_input",
    )
    top_k = st.number_input(_t(language, "top_k"), min_value=1, max_value=20, value=5, step=1)
    submitted = st.button(_t(language, "submit"), type="primary")

    if not submitted:
        st.info(_t(language, "idle_info"))
        return

    if not question.strip():
        st.error(_t(language, "missing_question"))
        return

    try:
        contextual_question = _build_contextual_question(question, conversation_history)
        route_decision = route_question(contextual_question)
        needs_paper, needs_repo = _source_requirements(route_decision.source_filter)

        if needs_paper and uploaded_pdf is None:
            st.error(_t(language, "missing_pdf"))
            return

        if needs_repo and not github_url.strip():
            st.error(_t(language, "missing_repo"))
            return

        pdf_hash = ""
        text_quality = None
        paper_chunks = []
        if needs_paper and uploaded_pdf is not None:
            pdf_path = _save_uploaded_pdf(uploaded_pdf)
            pdf_hash = hash_file(pdf_path)
            text_quality = assess_pdf_text_quality(pdf_path)
            paper_chunks = parse_pdf(pdf_path)

        repo_scan = None
        repo_chunks = []
        normalized_repo_url = ""
        if needs_repo:
            normalized_repo_url = validate_github_url(github_url)
            repo_scan = scan_repository_with_metadata(normalized_repo_url)
            repo_chunks = repo_scan.chunks

        dataset_id = build_dataset_id(
            pdf_hash=pdf_hash or f"no-pdf:{route_decision.source_filter}",
            normalized_repo_url=(repo_scan.repo_url if repo_scan else f"no-repo:{route_decision.source_filter}"),
            commit_hash=repo_scan.commit_hash if repo_scan else None,
        )
        index_result = build_index(
            paper_chunks=paper_chunks,
            repo_chunks=repo_chunks,
            dataset_id=dataset_id,
            repo_url=repo_scan.repo_url if repo_scan else "",
            repo_commit_hash=repo_scan.commit_hash if repo_scan else None,
            clear_existing=True,
        )
        retrieved_chunks = retrieve_evidence(
            question=contextual_question,
            dataset_id=index_result.dataset_id,
            source_filter=route_decision.source_filter,
            top_k=int(top_k),
        )
        scanned_file_paths = set(repo_scan.scanned_files) if repo_scan else set()
        valid_chunks = filter_valid_evidence(
            retrieved_chunks=retrieved_chunks,
            dataset_id=index_result.dataset_id,
            repo_url=repo_scan.repo_url if repo_scan else "",
            scanned_file_paths=scanned_file_paths,
        )
        agent_answer = generate_constrained_answer(
            question=contextual_question,
            route_decision=route_decision,
            retrieved_chunks=valid_chunks,
            language=language,
        )
        st.session_state["last_dataset_id"] = index_result.dataset_id
        st.session_state["conversation_history"] = _append_conversation_turn(
            conversation_history,
            question,
            contextual_question,
            agent_answer,
            route_decision.question_type,
            route_decision.source_filter,
        )
    except ValueError as exc:
        logger.warning("User input validation failed: %s", exc)
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Unexpected application error")
        st.error(_t(language, "unexpected_error"))
        return

    paper_pages = sorted(
        {
            page_num
            for chunk in paper_chunks
            for page_num in range(chunk.page_num, (chunk.page_end or chunk.page_num) + 1)
        }
    )
    repo_files = repo_scan.scanned_files if repo_scan else []
    key_files = [file_path for file_path in repo_files if is_key_file_path(file_path)]
    chunking_strategies = sorted({chunk.chunking_strategy for chunk in paper_chunks})

    st.success(_t(language, "run_success"))
    st.markdown(f"**{_t(language, 'answered_question')}**: {question.strip()}")
    if contextual_question.strip() != question.strip():
        with st.expander(_t(language, "contextual_question")):
            st.text(contextual_question)
    summary_metrics: list[tuple[str, object]] = [
        (_t(language, "index_chunks"), index_result.document_count),
        (_t(language, "dataset"), index_result.dataset_id[:8]),
        (_t(language, "valid_evidence_metric"), len(valid_chunks)),
    ]
    if pdf_hash:
        summary_metrics.insert(0, (_t(language, "parsed_pages"), len(paper_pages)))
    if repo_scan:
        summary_metrics.insert(1 if paper_chunks else 0, (_t(language, "scanned_files"), len(repo_files)))
        summary_metrics.insert(2 if paper_chunks else 1, (_t(language, "key_files_metric"), len(key_files)))
    _render_metric_grid(summary_metrics)

    if text_quality is not None:
        st.subheader(_t(language, "text_quality"))
        col1, col2, col3 = st.columns(3)
        col1.metric(_t(language, "extracted_pages"), f"{text_quality.extracted_pages}/{text_quality.page_count}")
        col2.metric(_t(language, "low_text_pages"), text_quality.low_text_pages)
        col3.metric(_t(language, "readable_ratio"), f"{text_quality.readable_char_ratio:.2f}")
        quality_message = _t(language, "ocr_recommended") if text_quality.needs_ocr else _t(language, "ocr_not_needed")
        quality_detail = (
            _t(language, "quality_ocr_detail")
            if text_quality.needs_ocr
            else _t(language, "quality_good_detail")
        )
        if text_quality.needs_ocr:
            st.warning(f"{quality_message}: {quality_detail}")
        else:
            st.info(f"{quality_message}: {quality_detail}")

    st.subheader(_t(language, "current_dataset"))
    if repo_scan:
        st.write(f"{_t(language, 'repository_url')}: `{repo_scan.repo_url}`")
        st.write(f"{_t(language, 'commit_hash')}: `{repo_scan.commit_hash or _t(language, 'unknown')}`")
    if pdf_hash:
        st.write(f"{_t(language, 'pdf_hash')}: `{pdf_hash[:12]}`")
    st.write(f"{_t(language, 'embedding_provider')}: `{os.getenv('EMBEDDING_PROVIDER', 'hash')}`")
    st.write(f"{_t(language, 'embedding_model')}: `{os.getenv('EMBEDDING_MODEL', '') or 'hash-local'}`")
    answer_provider = os.getenv("LLM_ANSWER_PROVIDER", "").strip() or os.getenv("LLM_ROUTER_PROVIDER", "").strip()
    answer_model = os.getenv("LLM_ANSWER_MODEL", "").strip() or os.getenv("LLM_ROUTER_MODEL", "").strip()
    st.write(f"{_t(language, 'answer_llm')}: `{answer_provider or 'disabled'}:{answer_model or 'disabled'}`")
    if chunking_strategies:
        st.write(f"{_t(language, 'chunking_strategy')}: `{', '.join(chunking_strategies)}`")

    if repo_scan:
        st.subheader(_t(language, "key_repository_files"))
        if key_files:
            for file_path in key_files:
                st.write(f"- `{file_path}`")
        else:
            st.caption(_t(language, "no_key_files"))

    st.subheader(_t(language, "agent_route"))
    col1, col2, col3 = st.columns(3)
    col1.metric(_t(language, "question_type"), _route_label(language, route_decision.question_type))
    col2.metric(_t(language, "search_source"), _route_label(language, route_decision.source_filter))
    col3.metric(_t(language, "route_confidence"), f"{route_decision.confidence:.2f}")
    st.caption(
        f"{_t(language, 'routing_method')}: {_route_label(language, route_decision.method)}. "
        f"{_route_rationale(language, route_decision.question_type, route_decision.rationale)}"
    )

    _render_agent_answer(agent_answer, language)

    st.subheader(_t(language, "top_valid_evidence"))
    if valid_chunks:
        for index, item in enumerate(valid_chunks, start=1):
            _render_retrieved_chunk(index, item, language)
    else:
        st.warning(_t(language, "no_route_evidence"))

    with st.expander(_t(language, "raw_results")):
        st.json([item.model_dump(mode="json") for item in retrieved_chunks])

    with st.expander(_t(language, "filtered_results")):
        st.json([item.model_dump(mode="json") for item in valid_chunks])

    with st.expander(_t(language, "structured_answer")):
        st.json(agent_answer.model_dump(mode="json"))


if __name__ == "__main__":
    main()
