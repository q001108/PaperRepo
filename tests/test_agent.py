import pytest

from src.answerer import generate_constrained_answer
from src.router import route_question
from src.schemas import RetrievedChunk, RouteDecision


TEST_QUESTIONS = [
    ("What method does the paper propose?", "paper_question"),
    ("What experiments are reported in the paper?", "paper_question"),
    ("论文的主要结论是什么？", "paper_question"),
    ("Which dependencies are listed in requirements.txt?", "repo_question"),
    ("Where is the main implementation code in the repository?", "repo_question"),
    ("仓库 README 说明了什么？", "repo_question"),
    ("Does the repository implement the method described in the paper?", "cross_source_check"),
    ("Compare the paper method with the GitHub code implementation.", "cross_source_check"),
    ("论文方法和仓库实现是否一致？", "cross_source_check"),
    ("Can the repository reproduce the paper experiments?", "cross_source_check"),
]


def _cross_source_decision() -> RouteDecision:
    return RouteDecision(
        question_type="cross_source_check",
        source_filter="both",
        confidence=0.85,
        rationale="cross route",
        method="rules",
    )


def _paper_chunk(content: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        metadata={"source_type": "paper", "page_num": 3, "chunk_id": "p1"},
        score=score,
        source_type="paper",
    )


def _repo_chunk(content: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        metadata={
            "source_type": "repo",
            "file_path": "src/retrieval.py",
            "line_start": 4,
            "line_end": 5,
            "chunk_id": "r1",
        },
        score=score,
        source_type="repo",
    )


@pytest.mark.parametrize(("question", "expected_type"), TEST_QUESTIONS)
def test_rule_router_covers_test_questions(question: str, expected_type: str):
    decision = route_question(question)

    assert decision.question_type == expected_type


def test_constrained_answer_is_missing_without_required_single_source_evidence():
    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.8,
        rationale="paper route",
        method="rules",
    )

    answer = generate_constrained_answer(
        question="What does the paper say?",
        route_decision=decision,
        retrieved_chunks=[],
    )

    assert answer.status == "missing"
    assert "No unsupported conclusion" in answer.answer
    assert answer.paper_evidence == []
    assert answer.repo_evidence == []


def test_supported_status_requires_confidence_threshold():
    decision = RouteDecision(
        question_type="repo_question",
        source_filter="repo",
        confidence=0.8,
        rationale="repo route",
        method="rules",
    )
    answer = generate_constrained_answer(
        question="Where is the implementation?",
        route_decision=decision,
        retrieved_chunks=[
            RetrievedChunk(
                content="implementation candidate",
                metadata={
                    "source_type": "repo",
                    "file_path": "src/main.py",
                    "line_start": 1,
                    "line_end": 1,
                    "dataset_id": "dataset",
                    "repo_url": "https://github.com/example/repo",
                },
                score=0.0,
                source_type="repo",
            )
        ],
    )

    assert not (answer.status == "supported" and answer.confidence == 0.0)
    assert answer.status != "supported"


def test_cross_source_missing_paper_evidence_is_insufficient_zero_confidence():
    answer = generate_constrained_answer(
        question="Does the repository implement the paper method?",
        route_decision=_cross_source_decision(),
        retrieved_chunks=[
            _repo_chunk("The repository implements the retrieval module.", score=0.9)
        ],
    )

    assert answer.status == "insufficient"
    assert answer.confidence == 0.0
    assert "paper evidence" in answer.answer
    assert answer.paper_evidence == []
    assert answer.repo_evidence


def test_cross_source_missing_repo_evidence_is_insufficient_zero_confidence():
    answer = generate_constrained_answer(
        question="Does the repository implement the paper method?",
        route_decision=_cross_source_decision(),
        retrieved_chunks=[
            _paper_chunk("The paper proposes a retrieval module.", score=0.9)
        ],
    )

    assert answer.status == "insufficient"
    assert answer.confidence == 0.0
    assert "repo evidence" in answer.answer
    assert answer.paper_evidence
    assert answer.repo_evidence == []


def test_cross_source_consistent_evidence_is_supported():
    answer = generate_constrained_answer(
        question="Does the repository implement the paper method?",
        route_decision=_cross_source_decision(),
        retrieved_chunks=[
            _paper_chunk("The paper proposes and uses a retrieval module.", score=0.9),
            _repo_chunk("The repository implements the retrieval module.", score=0.9),
        ],
    )

    assert answer.status == "supported"
    assert answer.confidence >= 0.60
    assert "[P1]" in answer.answer
    assert "[R1]" in answer.answer


def test_cross_source_opposite_evidence_is_conflict():
    answer = generate_constrained_answer(
        question="Does the repository implement the paper method?",
        route_decision=_cross_source_decision(),
        retrieved_chunks=[
            _paper_chunk("The paper proposes and uses a retrieval module.", score=0.9),
            _repo_chunk("The repository does not implement the retrieval module.", score=0.9),
        ],
    )

    assert answer.status == "conflict"
    assert answer.confidence >= 0.60
    assert "[P1]" in answer.answer
    assert "[R1]" in answer.answer
