from app import _build_contextual_question, _source_requirements


def test_source_requirements_follow_route_source_filter():
    assert _source_requirements("repo") == (False, True)
    assert _source_requirements("paper") == (True, False)
    assert _source_requirements("both") == (True, True)


def test_contextual_question_is_unchanged_without_history():
    assert _build_contextual_question("这些代码对应哪些模块？", []) == "这些代码对应哪些模块？"


def test_contextual_question_includes_recent_turns_for_follow_up():
    contextual_question = _build_contextual_question(
        "这些代码对应哪些模块？",
        [
            {
                "question": "这个仓库的 README 主要说明了什么？",
                "answer": "README 说明这是 SGRE 的官方 PyTorch 实现，并提供训练脚本。",
                "question_type": "repo_question",
                "source_filter": "repo",
            }
        ],
    )

    assert "Current question: 这些代码对应哪些模块？" in contextual_question
    assert "README" in contextual_question
    assert "SGRE" in contextual_question
    assert "repo_question / repo" in contextual_question
