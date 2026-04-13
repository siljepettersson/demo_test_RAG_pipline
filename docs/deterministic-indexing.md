# Deterministic Indexing

This note documents the changes that make indexing behavior more repeatable across runs.

## What Is Stabilized

The pipeline now keeps these steps deterministic where practical:
- document load order
- chunk ordering
- chunk identifiers
- vector store insertion identifiers

## How It Works

- source documents are sorted by resolved path before metadata is attached
- chunked documents are sorted by `chunk_id`
- each chunk gets a stable `chunk_id` based on source path and chunk index
- Chroma documents are written with explicit IDs derived from `chunk_id`

Example chunk ID:

```text
fjordmat/kampanjebrief-sommermeny-2025.md#chunk-000
```

## Verification Commands

Build and inspect the pipeline:

```bash
uv run python src/rag_pipeline.py --summary --metadata-audit --skip-queries
```

Check that two independent load/chunk runs produce the same chunk IDs:

```bash
uv run python -c "from src.rag_pipeline import load_documents, chunk_documents; run1=[chunk.metadata['chunk_id'] for chunk in chunk_documents(load_documents())]; run2=[chunk.metadata['chunk_id'] for chunk in chunk_documents(load_documents())]; print(run1 == run2); print(len(run1), len(set(run1)))"
```

Expected output on the current dataset:
- `True`
- `99 99`

## Why This Matters

This makes the MVP easier to debug and explain:
- metadata audits are easier to compare over time
- retrieval investigations can refer to stable chunk IDs
- later ranking changes are easier to evaluate because the indexing baseline is steadier
