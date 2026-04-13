# AGENTS.md — Project Guide for AI Assistants

## Purpose

This repository is a demo/MVP for a job application for an "AI-spesialist" role at a marketing agency group.

The assignment has three parts:
1. Explain what AI capability should be prioritized first at the agency, and why.
2. Build a simple MVP/demo/POC that demonstrates the idea.
3. Write briefly about the approach, choices made, what would be improved with more time, and time spent.

This is not only a coding exercise. It is a product-judgment exercise.

## Product Thesis

Build a RAG-based Client Knowledge Base: an internal assistant where agency employees can ask questions about client brand guidelines, campaigns, reports, strategies, and meeting notes, and receive grounded answers with source references.

This is a strong first AI use case for a marketing agency because it is:
- useful across multiple teams
- easy to explain to non-technical stakeholders
- realistic to prototype quickly
- a strong example of practical AI value rather than novelty

The core argument should stay explicit throughout the project:
- Retrieval quality matters more than a flashy interface.
- Source-grounded answers matter more than broad chatbot behavior.
- A simple, reliable internal knowledge assistant is a more credible first build than a generic "AI copilot".

## Audience

The main audience is a hiring manager at a marketing agency.

The project should feel:
- practical
- thoughtful
- explainable
- scoped with good judgment

## MVP Goal

The MVP should prove four things:
- Norwegian client documents can be indexed and retrieved reliably.
- Users can ask natural questions and get relevant answers.
- Answers are tied clearly to source material.
- The system is simple to run and demo locally.

## Technical Direction

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | LangChain | Standard RAG tooling, fast to prototype |
| Vector DB | ChromaDB | Local, lightweight, no server needed |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Local, free, good multilingual support including Norwegian |
| UI | Streamlit | Fastest way to build a clean interactive demo |
| LLM | TBD for UI phase | Only needed for answer generation in the app |
| Language | Python | Entire pipeline can stay in one language |

## Key Architectural Principle

The retrieval pipeline should work without API keys.

Document loading, chunking, embedding, and vector storage should run locally. This keeps the core demo:
- cheap
- reproducible
- easy to test
- easy to explain

If an external LLM is used in the Streamlit app, treat it as a thin answer-generation layer on top of a retrieval system that already works on its own.

## Fictional Client Dataset

There are four fictional Norwegian companies with four documents each, for a total of sixteen documents.

| Client | Type | Documents |
|--------|------|-----------|
| **Fjordmat** | Restaurant chain | Brand guidelines, summer campaign brief, christmas campaign report, Q1 meeting notes |
| **Spareklar** | Fintech app | Brand guidelines, feature launch brief, Google Ads Q4 report, social media strategy |
| **Nordvik** | Sustainable clothing | Brand guidelines, autumn collection brief, influencer strategy, Instagram campaign report |
| **Skytjenester AS** | B2B SaaS | Brand guidelines, lead gen brief, SEO report, customer case study |

All documents are written in Norwegian and include realistic details such as budgets, KPIs, timelines, action items, contact people, and campaign results.

## Repository Layout

```text
data/
  fjordmat/
  nordvik/
  skytjenester/
  spareklar/
src/
  rag_pipeline.py
.gitignore
README.md
pyproject.toml
uv.lock
```

Main files:
- `data/`: fictional client documents used for retrieval
- `src/rag_pipeline.py`: indexing and retrieval prototype
- `pyproject.toml`: project metadata and dependencies
- `uv.lock`: locked dependencies for reproducibility
- `README.md`: final write-up and usage guidance

Generated local artifacts:
- `vectorstore/`: Chroma persistence created after indexing; should not be committed

## Phase Status

### Phase 1: Client Data
Status: complete

- 16 fictional Norwegian documents exist under `data/`

### Phase 2: RAG Pipeline
Status: in progress

Current state:
- `src/rag_pipeline.py` exists
- the code can load documents, chunk them, embed them, persist a Chroma vector store, and run sample queries
- the pipeline runs with `uv run python src/rag_pipeline.py`

Current priorities in this phase:
- improve retrieval quality and ranking consistency
- keep metadata clean for source attribution
- make indexing deterministic
- keep the code simple and easy to explain

### Phase 3: Streamlit UI
Status: next

Target MVP:
- chat-style interface
- question input
- answer generation based on retrieved chunks
- visible source documents for each answer
- optional client filter in the sidebar

Important principle:
- Do not overbuild the UI.
- The UI exists to demonstrate grounded retrieval, not to compete with production chat tools.

### Phase 4: Write-up
Status: last

The write-up should cover:
- what was built
- why this was the right first AI use case
- architecture and technical choices
- limitations
- what would be improved with more time
- approximate time spent

## Priority Order

If there is a tradeoff, prioritize in this order:
1. Retrieval quality
2. Source attribution
3. Simple and clear demo UX
4. Nice-to-have UI polish

The project will be judged more on product thinking and solid RAG fundamentals than on advanced architecture or visual complexity.

## Operational Rules

### Language

- Documents are in Norwegian.
- Code, comments, and technical explanations should be in English.
- Demo questions and retrieval evaluation should primarily use Norwegian.

### Development

- Use `uv` for dependency management.
- Python version is `>=3.11`.
- Prefer simple, explicit code over abstraction.
- Keep the codebase easy to explain to a non-technical stakeholder.
- Do not over-engineer this MVP.

### Git

- Never push directly to `main`.
- Prefer feature branches and small, descriptive commits.

## Retrieval Requirements

When working on the pipeline, optimize for explainable retrieval, not hidden magic.

At a minimum, each retrieved chunk should be traceable back to:
- client
- document title or filename
- document type
- source path
- chunk index or equivalent chunk identifier

If metadata changes are introduced, preserve or improve source attribution quality.

Indexing should be deterministic where practical:
- stable document ordering
- stable chunking behavior
- stable metadata fields
- no unnecessary randomness in retrieval setup

## Evaluation Guidance

Use realistic Norwegian questions to judge whether the system is working. Questions should cover different clients and document types.

Suggested evaluation questions:
- "Hva er Fjordmats tone of voice?"
- "Hva var ROAS for Spareklars Google Ads i Q4 2024?"
- "Hvilke influencere samarbeider Nordvik med?"
- "Hva sier kundecaset om Skytjenester sine resultater hos LogistikkPartner?"

When evaluating retrieval, prioritize:
- correct client
- correct document type
- relevant chunk content
- clear source metadata

A retrieval change is valuable only if it improves one or more of those dimensions without making the system meaningfully harder to explain.

## LLM Guidance For Phase 3

The LLM choice is open. Choose based on practicality, not ideology.

Decision criteria:
- simple setup for demo purposes
- reliable answer quality on Norwegian input
- easy prompt control
- acceptable cost for a small MVP
- clear separation between retrieval and generation

Reasonable approach:
- keep retrieval provider-independent
- add one LLM option for the demo app
- avoid building provider abstraction unless there is a real need

Do not use the LLM to hide weak retrieval. If retrieval is wrong, fix retrieval first.

## Definition Of Done By Phase

### RAG Pipeline

The pipeline is good enough when:
- indexing works locally from the provided data
- a small set of realistic Norwegian questions retrieves the correct material
- source metadata is visible and trustworthy
- the flow is simple enough to explain in a short demo

### Streamlit MVP

The app is good enough when:
- a user can ask a question in natural language
- the app returns an answer grounded in retrieved content
- sources are shown clearly
- the UX is clean and minimal, without unnecessary features

### Final Write-up

The write-up is good enough when:
- it explains why this was the right first AI use case
- it describes the architecture in plain language
- it is honest about limitations
- it shows practical judgment about scope and next steps

## Commands

Set up dependencies:

```bash
uv sync
```

Run the retrieval pipeline:

```bash
uv run python src/rag_pipeline.py
```

Run the app once Phase 3 exists:

```bash
uv run streamlit run app.py
```

## What To Avoid

- Do not turn this into a generic chatbot.
- Do not spend time on unnecessary framework complexity.
- Do not hide weak retrieval behind confident LLM output.
- Do not optimize for production-scale concerns that are irrelevant to a hiring-demo MVP.
- Do not let UI work consume time that should have gone to retrieval quality.

## Final Reminder

The strongest version of this project says:

"Here is the first AI capability I would prioritize at a marketing agency, here is why it matters, and here is a credible MVP that demonstrates it."
