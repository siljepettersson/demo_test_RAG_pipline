# Client Knowledge Base for a Marketing Agency

This repository is a small MVP for a proposed first AI capability at a marketing agency: an internal, source-grounded assistant that helps employees find answers in client material such as brand guidelines, campaign briefs, reports, strategies, and meeting notes.

The core idea is simple:

Instead of starting with a generic "AI copilot", start with a practical internal knowledge assistant that solves a real problem across teams and is easy to explain, demo, and trust.

## 1. What AI capability should be prioritized first, and why?

I would prioritize a RAG-based client knowledge base first.

Why this is the right first move:

- It solves a broad internal problem. Account managers, strategists, paid specialists, content teams, and leadership all need fast access to client-specific knowledge.
- It is immediately useful. Agencies already have scattered information in briefs, reports, notes, and guidelines. The problem is usually access and reuse, not lack of data.
- It is easy to explain to non-technical stakeholders. "Ask a question, get an answer with sources" is much easier to trust than a vague promise of "AI transformation".
- It creates value without requiring deep process change. Teams do not need to change how they work dramatically to benefit from it.
- It is a credible MVP. A local retrieval pipeline with transparent sources is realistic to build quickly and still demonstrates sound product judgment.

The important product choice is to optimize for retrieval quality and source attribution before optimizing for a polished chat experience. If the retrieved context is wrong, a fluent answer does not help.

## 2. What was built

The MVP is a local RAG demo for fictional Norwegian clients.

It includes:

- 16 fictional client documents across 4 clients
- a retrieval pipeline that loads, chunks, embeds, and indexes documents locally
- metadata normalization for reliable source attribution
- deterministic chunk IDs and indexing behavior where practical
- a lightweight retrieval tuning layer that uses explicit client and document-type hints from the question
- a simple Streamlit interface for asking questions and inspecting sources
- optional answer generation through an external LLM provider on top of the retrieval layer

The retrieval system works without API keys. That was a deliberate choice so the core of the demo stays cheap, reproducible, and easy to explain.

## 3. Why this use case fits a marketing agency

This use case fits agency work particularly well because client knowledge is:

- high-value
- distributed across many documents
- reused by many roles
- easy to get wrong when people rely on memory

Typical questions this system should help answer:

- "Hva er Fjordmats tone of voice?"
- "Hva var ROAS for Spareklars Google Ads i Q4 2024?"
- "Hvilke influencere samarbeider Nordvik med?"
- "Hva sier kundecaset om Skytjenester sine resultater hos LogistikkPartner?"

This is a better first build than a generic chatbot because it stays narrow, grounded, and auditable.

## 4. Architecture and technical choices

The project uses a simple stack chosen for explainability and speed:

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python | Fastest way to keep the full pipeline in one language |
| Retrieval tooling | LangChain | Standard, familiar RAG tooling for a quick MVP |
| Vector store | ChromaDB | Local persistence, lightweight, easy to run |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Local, multilingual, suitable for Norwegian |
| UI | Streamlit | Fastest path to a clean demo interface |
| LLM layer | Optional | Retrieval remains useful even without external generation |

## 5. How the pipeline works

The retrieval pipeline lives in [src/rag_pipeline.py](/home/silje/project/demo_test_RAG_pipline/src/rag_pipeline.py).

It works in five steps:

1. Load markdown documents from `data/` and sort them deterministically.
2. Attach normalized metadata to each document:
   `client`, `filename`, `document_title`, `document_type`, `source`, `source_path`
3. Split documents into overlapping chunks and assign stable chunk metadata:
   `chunk_index`, `chunk_id`
4. Embed chunks locally and persist them in Chroma using stable IDs.
5. Retrieve relevant chunks, then lightly rerank them using query hints such as explicit client names and likely document types.

This is intentionally simple. There is no heavy reranker, no complicated orchestration layer, and no hidden LLM dependency in the retrieval path.

## 6. Source attribution and explainability

One of the main priorities in this project was making retrieval explainable.

Each retrieved chunk can be traced back to:

- client
- document title
- document type
- source file
- chunk index / chunk ID

That matters because this MVP is not trying to impress with broad chat behavior. It is trying to show trustworthy retrieval and grounded answers.

## 7. Current MVP status

What is working now:

- local indexing from the provided Norwegian markdown corpus
- deterministic document and chunk ordering
- clean source metadata for attribution
- local Chroma persistence
- retrieval over the fictional client dataset
- a Streamlit app with question input, optional client filtering, source display, and index rebuild
- optional LLM answer generation through configured API keys

Current retrieval evaluation result:

```text
Top-1 source matches: 4/4
```

That evaluation covers one representative question per client and checks expected client, expected document type, and expected source file.

## 8. How the retrieval quality was tested

The project includes a lightweight built-in retrieval evaluation loop.

Run:

```bash
uv run python src/rag_pipeline.py --evaluate --skip-queries
```

This:

- loads and chunks the corpus
- rebuilds the vector store
- runs fixed Norwegian evaluation questions
- reports whether top-1 retrieval matches the expected source

This is intentionally modest, but useful for an MVP. It makes retrieval changes easier to validate without adding much complexity.

## 9. Streamlit demo

The app lives in [app.py](/home/silje/project/demo_test_RAG_pipline/app.py).

The UI is intentionally minimal:

- ask a question in Norwegian
- optionally filter by client
- retrieve the most relevant chunks
- generate a short answer
- show the source documents used
- let the user expand a source to inspect the retrieved chunk text

This keeps the demo focused on grounded retrieval instead of trying to mimic a full chat product.

## 10. Project structure

```text
data/
  fjordmat/
  nordvik/
  skytjenester/
  spareklar/
src/
  rag_pipeline.py
  llm.py
app.py
README.md
pyproject.toml
uv.lock
```

## 11. How to run locally

Install dependencies:

```bash
uv sync
```

Run the retrieval pipeline:

```bash
uv run python src/rag_pipeline.py
```

Run retrieval evaluation:

```bash
uv run python src/rag_pipeline.py --evaluate --skip-queries
```

Run the Streamlit app:

```bash
uv run streamlit run app.py
```

## 12. Environment variables

The retrieval pipeline itself does not require API keys.

If you want answer generation in the Streamlit app, add one of these to `.env`:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MINIMAX_API_KEY=...
```

If no LLM provider is configured, the app still works in retrieval-only mode and shows the top supporting source text.

## 13. Limitations

This is still an MVP, and the limits are important to state clearly:

- The dataset is small and fictional.
- The retrieval evaluation set is still small.
- Query understanding is rule-based and intentionally lightweight.
- There is no authentication, permissions model, or production document sync.
- The UI is intentionally minimal and not production-grade.
- The LLM layer is thin and should not be used to hide weak retrieval.

## 14. What I would improve with more time

If I had more time, I would prioritize:

- a broader retrieval test set with paraphrased and harder Norwegian questions
- a small `pytest` suite for metadata, determinism, and retrieval expectations
- more robust ranking for ambiguous questions that do not explicitly mention a client
- clearer answer formatting in the UI, especially around uncertainty and source usage
- import/sync flows for more realistic agency document formats
- a simple per-client filter or permissions model closer to how agencies actually work

## 15. Why this is a credible first AI build

The strongest version of this project is not "look at this chatbot".

It is:

"Here is one AI capability a marketing agency can realistically prioritize first. It is useful across teams, explainable to stakeholders, grounded in source material, and practical to prototype quickly."

That is the standard this MVP was designed around.

## 16. Time spent

To keep this write-up honest, I would fill in the final submitted time based on actual work log rather than guess here.

Suggested line for submission:

```text
Approximate time spent: [replace with your real number] hours
```
