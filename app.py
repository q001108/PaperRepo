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
from src.pdf_parser import parse_pdf
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


def _save_uploaded_pdf(uploaded_file) -> Path:
    upload_dir = Path(".uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_path = upload_dir / safe_name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _render_retrieved_chunk(index: int, item: RetrievedChunk) -> None:
    metadata = item.metadata
    score = f"{item.score:.3f}" if item.score is not None else "N/A"

    if item.source_type == "paper":
        page_num = metadata.get("page_num", "N/A")
        section_title = metadata.get("section_title") or "Unknown section"
        st.markdown(f"**{index}. Paper evidence** | page `{page_num}` | score `{score}`")
        st.caption(f"Section: {section_title}")
        st.text_area(
            "Paper snippet",
            item.content,
            height=160,
            key=f"paper-evidence-{index}",
            disabled=True,
        )
        return

    file_path = metadata.get("file_path", "N/A")
    line_start = metadata.get("line_start", "N/A")
    line_end = metadata.get("line_end", "N/A")
    st.markdown(
        f"**{index}. Repo evidence** | `{file_path}` lines `{line_start}-{line_end}` | score `{score}`"
    )
    st.code(item.content, language=str(metadata.get("language") or "text").lower())


def _render_answer_evidence(item: AnswerEvidence) -> None:
    metadata = item.metadata
    score = f"{item.score:.3f}" if item.score is not None else "N/A"

    if item.source_type == "paper":
        page_num = metadata.get("page_num", "N/A")
        section_title = metadata.get("section_title") or "Unknown section"
        st.markdown(f"**{item.evidence_id}** | paper page `{page_num}` | score `{score}`")
        st.caption(f"Section: {section_title}")
        st.text_area(
            f"Paper evidence {item.evidence_id}",
            item.content,
            height=150,
            key=f"answer-paper-{item.evidence_id}",
            disabled=True,
        )
        return

    file_path = metadata.get("file_path", "N/A")
    line_start = metadata.get("line_start", "N/A")
    line_end = metadata.get("line_end", "N/A")
    st.markdown(
        f"**{item.evidence_id}** | repo `{file_path}` lines `{line_start}-{line_end}` | score `{score}`"
    )
    st.code(item.content, language=str(metadata.get("language") or "text").lower())


def _render_agent_answer(answer: AgentAnswer) -> None:
    st.subheader("Agent answer")
    col1, col2 = st.columns(2)
    col1.metric("Status", answer.status)
    col2.metric("Confidence", f"{answer.confidence:.2f}")
    st.write(answer.answer)

    st.subheader("Bound evidence")
    if answer.paper_evidence:
        st.markdown("**Paper evidence**")
        for item in answer.paper_evidence:
            _render_answer_evidence(item)
    if answer.repo_evidence:
        st.markdown("**Repository evidence**")
        for item in answer.repo_evidence:
            _render_answer_evidence(item)
    if not answer.paper_evidence and not answer.repo_evidence:
        st.warning("No evidence was strong enough for a supported answer.")

    st.subheader("Limitations")
    for limitation in answer.limitations:
        st.write(f"- {limitation}")


def main() -> None:
    st.set_page_config(
        page_title="PaperRepo Evidence-RAG Agent",
        page_icon="PR",
        layout="centered",
    )

    st.title("PaperRepo Evidence-RAG Agent")
    st.caption("Upload a paper, add a public GitHub repository, and ask an evidence-bound question.")

    last_dataset_id = st.session_state.get("last_dataset_id")
    if last_dataset_id:
        st.caption(f"Current dataset: `{last_dataset_id[:8]}`")
        if st.button("Clear current index"):
            try:
                clear_dataset_index(last_dataset_id)
                st.success(f"Cleared index for dataset `{last_dataset_id[:8]}`.")
            except Exception:
                logger.exception("Failed to clear current index")
                st.error("Failed to clear the current index. Check the logs for details.")

    with st.form("paper_repo_query_form"):
        uploaded_pdf = st.file_uploader("Paper PDF", type=["pdf"])
        github_url = st.text_input(
            "Public GitHub repository URL",
            placeholder="https://github.com/owner/repository",
        )
        question = st.text_area(
            "Question",
            placeholder="Does the repository implement the method described in the paper?",
            height=120,
        )
        top_k = st.number_input("Top-K evidence", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Rebuild Index and Run Agent")

    if not submitted:
        st.info("The Agent will route the question, retrieve evidence, and answer only when evidence is available.")
        return

    if uploaded_pdf is None:
        st.error("Please upload a PDF file.")
        return

    if not github_url.strip():
        st.error("Please enter a GitHub repository URL.")
        return

    if not question.strip():
        st.error("Please enter a question.")
        return

    try:
        normalized_repo_url = validate_github_url(github_url)
        route_decision = route_question(question)
        pdf_path = _save_uploaded_pdf(uploaded_pdf)
        pdf_hash = hash_file(pdf_path)
        paper_chunks = parse_pdf(pdf_path)
        repo_scan = scan_repository_with_metadata(normalized_repo_url)
        repo_chunks = repo_scan.chunks
        dataset_id = build_dataset_id(
            pdf_hash=pdf_hash,
            normalized_repo_url=repo_scan.repo_url,
            commit_hash=repo_scan.commit_hash,
        )
        index_result = build_index(
            paper_chunks=paper_chunks,
            repo_chunks=repo_chunks,
            dataset_id=dataset_id,
            repo_url=repo_scan.repo_url,
            repo_commit_hash=repo_scan.commit_hash,
            clear_existing=True,
        )
        retrieved_chunks = retrieve_evidence(
            question=question,
            dataset_id=index_result.dataset_id,
            source_filter=route_decision.source_filter,
            top_k=int(top_k),
        )
        scanned_file_paths = set(repo_scan.scanned_files)
        valid_chunks = filter_valid_evidence(
            retrieved_chunks=retrieved_chunks,
            dataset_id=index_result.dataset_id,
            repo_url=repo_scan.repo_url,
            scanned_file_paths=scanned_file_paths,
        )
        agent_answer = generate_constrained_answer(
            question=question,
            route_decision=route_decision,
            retrieved_chunks=valid_chunks,
        )
        st.session_state["last_dataset_id"] = index_result.dataset_id
    except ValueError as exc:
        logger.warning("User input validation failed: %s", exc)
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Unexpected application error")
        st.error("Something went wrong while preparing the request. Check the logs for details.")
        return

    paper_pages = sorted({chunk.page_num for chunk in paper_chunks})
    repo_files = repo_scan.scanned_files
    key_files = [file_path for file_path in repo_files if is_key_file_path(file_path)]

    st.success("Inputs accepted. Agent routing, retrieval, and constrained answering completed.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Parsed PDF pages", len(paper_pages))
    col2.metric("Scanned files", len(repo_files))
    col3.metric("Index chunks", index_result.document_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Key files", len(key_files))
    col2.metric("Dataset", index_result.dataset_id[:8])
    col3.metric("Valid evidence", len(valid_chunks))

    st.subheader("Current dataset")
    st.write(f"Repository URL: `{repo_scan.repo_url}`")
    st.write(f"Commit hash: `{repo_scan.commit_hash or 'unknown'}`")
    st.write(f"PDF hash: `{pdf_hash[:12]}`")

    st.subheader("Key repository files")
    if key_files:
        for file_path in key_files:
            st.write(f"- `{file_path}`")
    else:
        st.caption("No README, requirements.txt, environment.yml, or Dockerfile was found in the scanned files.")

    st.subheader("Agent route")
    col1, col2, col3 = st.columns(3)
    col1.metric("Question type", route_decision.question_type)
    col2.metric("Search source", route_decision.source_filter)
    col3.metric("Route confidence", f"{route_decision.confidence:.2f}")
    st.caption(f"Routing method: {route_decision.method}. {route_decision.rationale}")

    _render_agent_answer(agent_answer)

    st.subheader("Top-K valid retrieved evidence")
    if valid_chunks:
        for index, item in enumerate(valid_chunks, start=1):
            _render_retrieved_chunk(index, item)
    else:
        st.warning("No evidence matched the selected route.")

    with st.expander("Raw retrieval results"):
        st.json([item.model_dump(mode="json") for item in retrieved_chunks])

    with st.expander("Filtered valid retrieval results"):
        st.json([item.model_dump(mode="json") for item in valid_chunks])

    with st.expander("Structured agent answer"):
        st.json(agent_answer.model_dump(mode="json"))


if __name__ == "__main__":
    main()
