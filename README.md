# PaperRepo Evidence-RAG Agent

PaperRepo Evidence-RAG Agent is a minimal Streamlit demo for checking evidence across a research paper PDF and a public GitHub repository.

The project is built for an early-stage workflow: upload one paper, provide one repository URL, ask a question, and inspect the retrieved evidence before any full LLM answer generation is introduced.

It targets the scenario of information verification for AI paper reproduction, using a dual-source Evidence-RAG Agent over paper PDFs and GitHub repositories.

## Project Background

Research code repositories often claim to implement a paper, but validating that claim usually requires reading both the manuscript and the code. This demo creates a small, local evidence pipeline that can parse both sources, retrieve relevant snippets, and produce a conservative answer that is bound to visible evidence.

## Functional Boundary

Implemented:

- Parse uploaded PDFs with PyMuPDF into page-aware chunks.
- Clone public GitHub repositories with GitPython.
- Statically scan repository files only; no repository scripts are executed.
- Store paper and repository chunks in one local Chroma vector collection.
- Generate a deterministic `dataset_id` from the PDF hash, normalized GitHub URL, and Git commit hash.
- Filter retrieval by the current `dataset_id` to prevent historical index contamination.
- Validate repository evidence against the current repository URL and scanned file list.
- Preserve `source_type` metadata for `paper` and `repo` chunks.
- Route questions as `paper_question`, `repo_question`, or `cross_source_check`.
- Retrieve by source filter: paper, repo, or both.
- Produce structured, evidence-constrained answers.
- Return `insufficient` when required evidence is missing.

Out of scope for this demo:

- Running repository code, tests, training scripts, notebooks, or package installers.
- Proving that a repository fully reproduces a paper.
- Free-form LLM synthesis beyond the optional router fallback.
- Production authentication, multi-user storage, or hosted deployment hardening.

## Architecture

Text architecture diagram:

```text
Streamlit UI
  -> PDF upload -> PyMuPDF parser -> PaperChunk[]
  -> GitHub URL -> GitPython clone -> static repo scanner -> RepoChunk[]
  -> dataset_id = hash(PDF hash + normalized repo URL + commit hash)
  -> Chroma indexer clears and rebuilds chunks for that dataset_id
  -> Rule-first router chooses paper, repo, or both
  -> Retriever queries Chroma with dataset_id and source_type metadata filtering
  -> Evidence validator drops chunks from other repos or unknown file paths
  -> Evidence-constrained answerer returns AgentAnswer
  -> UI displays route, status, confidence, answer, evidence, and limitations
```

Core modules:

- `src/pdf_parser.py`: page-aware PDF text extraction.
- `src/repo_scanner.py`: static repository scanning with file-count and size limits.
- `src/embeddings.py`: configurable embedding function. The recommended local provider is `sentence_transformers`.
- `src/indexer.py`: Chroma persistence and chunk upsert.
- `src/retriever.py`: Top-K retrieval with `source_type` filtering.
- `src/dataset.py`: PDF hash, normalized GitHub URL, and `dataset_id` helpers.
- `src/evidence_validator.py`: current-repository evidence consistency checks.
- `src/router.py`: rule-first routing with optional LLM fallback.
- `src/answerer.py`: conservative answer generation bound to retrieved evidence.
- `src/schemas.py`: Pydantic models for chunks, route decisions, evidence, and answers.

## Requirements

- Python 3.11
- Git installed and available on PATH

## Installation

```powershell
cd E:\Demo\PaperRepo-Evidence-RAG-Agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

The recommended local embedding provider is `sentence_transformers`, which does not need an API key but downloads a local model on first use.

Configure runtime values in `.env`:

```text
OPENAI_API_KEY=
GITHUB_TOKEN=
LOG_LEVEL=INFO
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSIONS=384
SENTENCE_TRANSFORMERS_DEVICE=
CHROMA_PATH=.chroma
CHROMA_COLLECTION=paperrepo_evidence_minilm
LLM_ROUTER_PROVIDER=
LLM_ROUTER_MODEL=
```

API keys must stay in `.env`; do not place them in source files.

If the local model is too slow or you want to test only the software flow, switch back to:

```text
EMBEDDING_PROVIDER=hash
EMBEDDING_MODEL=
CHROMA_COLLECTION=paperrepo_evidence_hash
```

When changing embedding models, use a new `CHROMA_COLLECTION` or delete `.chroma/`, because Chroma collections expect a consistent vector dimension.

## Start

```powershell
streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

## Demo Flow

1. Upload a text-based PDF paper.
2. Enter a public GitHub repository URL.
3. Ask a question, for example: `Does the repository implement the method described in the paper?`
4. Choose the Top-K evidence count.
5. Run the Agent.
6. Review:
   - parsed PDF page count
   - scanned repository file count
   - current repository URL
   - current dataset ID prefix
   - current index chunk count
   - key repository files
   - route decision
   - answer status and confidence
   - paper evidence
   - repository evidence
   - limitations
   - raw structured JSON

More example questions are in `demo_questions.md`.

## Test

Run unit tests:

```powershell
pytest
```

Run a lightweight static check:

```powershell
python -m compileall app.py src tests
python -m pip check
```

## Known Limitations

- The MiniLM embedding model is lightweight and suitable for local demo evaluation, but stronger models may improve retrieval quality.
- PDF parsing depends on extractable text; scanned image PDFs need OCR, which is not implemented.
- Section title extraction is currently not implemented.
- Repository scanning is static and intentionally does not execute code.
- Cross-source answers are conservative and marked `partial` unless stronger verification is implemented.
- The optional LLM router uses the OpenAI chat completions endpoint only when configured.
- Chroma data, uploaded files, and cloned repositories are local working artifacts.
- Dataset isolation depends on metadata filtering and per-dataset cleanup, not separate Chroma databases.

## Security Notes

- `.env` and `.env.*` are ignored by Git, except `.env.example`.
- Uploaded files are written under `.uploads/`.
- Cloned repositories are written under `.repos/`.
- Chroma data is written under `.chroma/`.
- The scanner reads a limited file allowlist and never runs repository scripts.

## Project Structure

```text
app.py
demo_questions.md
src/
  pdf_parser.py
  repo_scanner.py
  indexer.py
  retriever.py
  router.py
  answerer.py
  embeddings.py
  schemas.py
tests/
requirements.txt
README.md
.env.example
.gitignore
```
