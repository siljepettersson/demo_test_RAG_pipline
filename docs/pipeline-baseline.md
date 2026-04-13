# Pipeline Baseline

This note captures the current baseline behavior of the retrieval pipeline before any retrieval-quality changes.

## Command

Run the baseline inspection with:

```bash
uv run python src/rag_pipeline.py --summary
```

This:
- loads the markdown corpus
- chunks the documents
- prints a compact corpus summary
- rebuilds the local Chroma vector store
- runs four fixed Norwegian test queries

## Baseline Snapshot

Observed on the current dataset:
- 16 documents loaded
- 99 chunks created
- 4 documents per client
- chunk distribution is uneven across clients because some source documents are much longer than others

Largest documents by chunk count:
- `spareklar/sosiale-medier-strategi-2025.md`: 11 chunks
- `nordvik/kampanjerapport-instagram-var-2025.md`: 8 chunks
- `skytjenester/seo-rapport-mars-2025.md`: 7 chunks
- `skytjenester/kundecase-logistikkpartner.md`: 7 chunks
- `fjordmat/motereferat-q1-2025.md`: 7 chunks

## Retrieval Observations

The current baseline is usable, but not yet clean:
- top-1 retrieval looks correct for the four sample questions
- top-2 retrieval is noisy for at least some questions
- the Fjordmat tone-of-voice question returns meeting notes above the brand guidelines document
- the Spareklar and Nordvik sample queries include an irrelevant second result from another client

## Why This Baseline Matters

This gives a fixed starting point for the next steps:
- audit metadata quality
- make indexing more deterministic
- improve ranking consistency for known Norwegian queries

The goal of later changes should be to improve retrieval quality without making the pipeline harder to explain.
