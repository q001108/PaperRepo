from src.schemas import AuditAnswer, Evidence, PaperChunk, RepoChunk


def test_core_schema_models_can_be_created():
    paper_chunk = PaperChunk(
        chunk_id="paper-1",
        content="Paper text",
        page_num=1,
        section_title="Introduction",
        source_file="paper.pdf",
    )
    repo_chunk = RepoChunk(
        chunk_id="repo-1",
        content="print('hello')",
        file_path="src/main.py",
        language="Python",
        line_start=1,
        line_end=10,
        repo_url="https://github.com/example/project",
    )
    evidence = Evidence(
        evidence_id="evidence-1",
        source_type="paper",
        source_id=paper_chunk.chunk_id,
        quote="Paper text",
        relevance="Matches the question.",
        score=0.8,
    )
    answer = AuditAnswer(
        question="How does it work?",
        answer="Placeholder",
        evidence=[evidence],
        limitations=[],
    )

    assert paper_chunk.chunk_id == "paper-1"
    assert str(repo_chunk.repo_url).startswith("https://github.com/example/project")
    assert answer.evidence[0].score == 0.8
