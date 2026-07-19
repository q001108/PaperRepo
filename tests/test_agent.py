import pytest

from src import llm_client
from src import router
from src.answerer import generate_constrained_answer
from src.router import route_question
from src.schemas import RetrievedChunk, RouteDecision


TEST_QUESTIONS = [
    ("What method does the paper propose?", "paper_question"),
    ("What experiments are reported in the paper?", "paper_question"),
    ("论文的主要结论是什么？", "paper_question"),
    ("这篇论文实现了什么方法", "paper_question"),
    ("论文提出了什么方法？", "paper_question"),
    ("这篇论文介绍了什么方法", "paper_question"),
    ("本文描述了什么方法", "paper_question"),
    ("Which dependencies are listed in requirements.txt?", "repo_question"),
    ("Where is the main implementation code in the repository?", "repo_question"),
    ("仓库 README 说明了什么？", "repo_question"),
    ("Does the repository implement the method described in the paper?", "cross_source_check"),
    ("Compare the paper method with the GitHub code implementation.", "cross_source_check"),
    ("论文方法和仓库实现是否一致？", "cross_source_check"),
    ("这个仓库是否实现了论文方法？", "cross_source_check"),
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


def test_deepseek_llm_router_uses_deepseek_api_config(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"question_type":"paper_question",'
                                '"confidence":0.77,'
                                '"rationale":"paper method question"}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("LLM_ROUTER_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ROUTER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    decision = router._llm_route("Ambiguous question")

    assert decision is not None
    assert decision.question_type == "paper_question"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "deepseek-v4-flash"


def test_deepseek_llm_answerer_synthesizes_final_answer(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer":"The paper proposes SGRE, an evidence-bound method summary [P1].",'
                                '"limitations":["Only provided evidence was used."]}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ANSWER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_ANSWER_CACHE_ENABLED", "false")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.9,
        rationale="paper method route",
        method="rules",
    )
    answer = generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=[
            RetrievedChunk(
                content="In this paper, we propose SGRE for robust label recognition.",
                metadata={
                    "source_type": "paper",
                    "page_num": 1,
                    "section_title": "Abstract",
                    "chunk_id": "p1",
                },
                score=0.9,
                source_type="paper",
            )
        ],
    )

    assert answer.answer.startswith("### Conclusion\n\nThe paper proposes SGRE, an evidence-bound method summary [P1].")
    assert "### Evidence" in answer.answer
    assert "- [P1] source=paper | page=1 | section=Abstract | score=0.900" in answer.answer
    assert "summary: In this paper, we propose SGRE for robust label recognition." in answer.answer
    assert answer.paper_evidence
    assert answer.repo_evidence == []
    assert any("deepseek:deepseek-v4-flash" in limitation for limitation in answer.limitations)
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    user_prompt = captured["json"]["messages"][1]["content"]
    assert user_prompt.index("Evidence:") < user_prompt.index("Question:")


def test_deepseek_llm_answerer_reuses_local_answer_cache(monkeypatch, tmp_path):
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer":"The paper proposes SGRE from the provided evidence [P1].",'
                                '"limitations":["Only provided evidence was used."]}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ANSWER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_ANSWER_CACHE_ENABLED", "true")
    monkeypatch.setenv("LLM_ANSWER_CACHE_PATH", str(tmp_path / "answers.json"))
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.9,
        rationale="paper method route",
        method="rules",
    )
    chunks = [
        RetrievedChunk(
            content="In this paper, we propose SGRE for robust label recognition.",
            metadata={
                "source_type": "paper",
                "page_num": 1,
                "section_title": "Abstract",
                "chunk_id": "p1",
                "dataset_id": "dataset-a",
            },
            score=0.9,
            source_type="paper",
        )
    ]

    first_answer = generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=chunks,
    )
    second_answer = generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=chunks,
    )

    assert call_count == 1
    assert first_answer.answer == second_answer.answer
    assert any("local LLM answer cache" in limitation for limitation in second_answer.limitations)


def test_llm_answer_cache_key_ignores_retrieval_score_changes(monkeypatch, tmp_path):
    call_count = 0

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"answer":"The paper proposes SGRE from stable evidence [P1].",'
                                '"limitations":["Only provided evidence was used."]}'
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_ANSWER_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_ANSWER_CACHE_ENABLED", "true")
    monkeypatch.setenv("LLM_ANSWER_CACHE_PATH", str(tmp_path / "answers.json"))
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.9,
        rationale="paper method route",
        method="rules",
    )

    def chunk(score: float) -> RetrievedChunk:
        return RetrievedChunk(
            content="In this paper, we propose SGRE for robust label recognition.",
            metadata={
                "source_type": "paper",
                "page_num": 1,
                "section_title": "Abstract",
                "chunk_id": "p1",
                "dataset_id": "dataset-a",
            },
            score=score,
            source_type="paper",
        )

    generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=[chunk(0.9)],
    )
    second_answer = generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=[chunk(0.7)],
    )

    assert call_count == 1
    assert "score=0.700" in second_answer.answer
    assert any("local LLM answer cache" in limitation for limitation in second_answer.limitations)


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


def test_paper_method_answer_summarizes_method_before_evidence(monkeypatch):
    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "")
    monkeypatch.setenv("LLM_ROUTER_PROVIDER", "")
    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.9,
        rationale="paper method route",
        method="rules",
    )

    answer = generate_constrained_answer(
        question="What method does the paper describe?",
        route_decision=decision,
        retrieved_chunks=[
            RetrievedChunk(
                content=(
                    "In this paper, we propose a semantic-guided representation enhancement framework. "
                    "The proposed framework consists of an inter-modal attention module and an intra-modal attention module."
                ),
                metadata={
                    "source_type": "paper",
                    "page_num": 1,
                    "page_end": 2,
                    "section_title": "Abstract",
                    "chunk_id": "p1",
                },
                score=0.9,
                source_type="paper",
            )
        ],
    )

    assert answer.status == "supported"
    assert answer.answer.startswith("### Conclusion")
    assert "method can be summarized as" in answer.answer
    assert "semantic-guided representation enhancement framework" in answer.answer
    assert "### Evidence" in answer.answer
    assert "source=paper | page=1-2 | section=Abstract | score=0.900" in answer.answer
    assert "[P1]" in answer.answer
    assert answer.follow_up_questions
    assert len(answer.follow_up_questions) <= 3
    assert "What modules make up the paper's method?" in answer.follow_up_questions


def test_chinese_paper_method_answer_summarizes_method_before_evidence(monkeypatch):
    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "")
    monkeypatch.setenv("LLM_ROUTER_PROVIDER", "")
    decision = RouteDecision(
        question_type="paper_question",
        source_filter="paper",
        confidence=0.9,
        rationale="paper method route",
        method="rules",
    )

    answer = generate_constrained_answer(
        question="这篇论文描述了什么方法？",
        route_decision=decision,
        retrieved_chunks=[
            RetrievedChunk(
                content="本文提出一种语义引导的表示增强框架，该框架包含跨模态注意力模块和模态内注意力模块。",
                metadata={
                    "source_type": "paper",
                    "page_num": 1,
                    "section_title": "Abstract",
                    "chunk_id": "p1",
                },
                score=0.9,
                source_type="paper",
            )
        ],
        language="zh",
    )

    assert answer.status == "supported"
    assert answer.answer.startswith("### 结论")
    assert "方法可以概括为" in answer.answer
    assert "语义引导的表示增强框架" in answer.answer
    assert "### 依据证据" in answer.answer
    assert "来源=论文 | 页码=1 | 章节=Abstract | 分数=0.900" in answer.answer
    assert "[P1]" in answer.answer


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
    assert "What evidence is currently missing?" in answer.follow_up_questions


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


def test_chinese_answer_templates_are_available(monkeypatch):
    monkeypatch.setenv("LLM_ANSWER_PROVIDER", "")
    monkeypatch.setenv("LLM_ROUTER_PROVIDER", "")
    answer = generate_constrained_answer(
        question="这个仓库是否实现了论文方法？",
        route_decision=_cross_source_decision(),
        retrieved_chunks=[
            _paper_chunk("The paper proposes and uses a retrieval module.", score=0.9),
            _repo_chunk("The repository implements the retrieval module.", score=0.9),
        ],
        language="zh",
    )

    assert answer.status == "supported"
    assert "跨来源证据" in answer.answer
    assert "结论仅限于" in answer.answer
    assert any("检索" in limitation for limitation in answer.limitations)
    assert 2 <= len(answer.follow_up_questions) <= 3
    assert any("论文" in question for question in answer.follow_up_questions)
    assert any("仓库" in question for question in answer.follow_up_questions)


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
