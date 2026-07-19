from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class PaperChunk(BaseModel):
    chunk_id: str = Field(..., description="Stable identifier for a paper text chunk.")
    content: str
    page_num: int = Field(..., ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    chunking_strategy: str = "section"
    source_type: Literal["paper"] = "paper"
    source_file: str | None = None


class PdfTextQuality(BaseModel):
    page_count: int = Field(..., ge=0)
    extracted_pages: int = Field(..., ge=0)
    low_text_pages: int = Field(..., ge=0)
    avg_chars_per_page: float = Field(..., ge=0)
    readable_char_ratio: float = Field(..., ge=0, le=1)
    needs_ocr: bool = False
    message: str


class RepoChunk(BaseModel):
    chunk_id: str = Field(..., description="Stable identifier for a repository chunk.")
    content: str
    file_path: str
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    source_type: Literal["repo"] = "repo"
    repo_url: HttpUrl | None = None
    language: str | None = None


class RepoScanResult(BaseModel):
    chunks: list[RepoChunk] = Field(default_factory=list)
    repo_url: str
    commit_hash: str | None = None
    scanned_files: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    dataset_id: str
    pdf_hash: str
    repo_url: str
    repo_commit_hash: str | None = None


class Evidence(BaseModel):
    evidence_id: str
    source_type: Literal["paper", "repo"]
    source_id: str
    quote: str
    relevance: str
    score: float | None = Field(default=None, ge=0, le=1)


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict[str, str | int | float | bool]
    score: float | None = None
    source_type: Literal["paper", "repo"]


class RouteDecision(BaseModel):
    question_type: Literal["paper_question", "repo_question", "cross_source_check"]
    source_filter: Literal["paper", "repo", "both"]
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    method: Literal["rules", "llm", "default"]


class AnswerEvidence(BaseModel):
    evidence_id: str
    source_type: Literal["paper", "repo"]
    content: str
    metadata: dict[str, str | int | float | bool]
    score: float | None = Field(default=None, ge=0, le=1)


class AgentAnswer(BaseModel):
    answer: str
    status: Literal["supported", "partial", "conflict", "missing", "insufficient"]
    confidence: float = Field(..., ge=0, le=1)
    paper_evidence: list[AnswerEvidence] = Field(default_factory=list)
    repo_evidence: list[AnswerEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class AuditAnswer(BaseModel):
    question: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
