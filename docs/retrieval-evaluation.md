# Retrieval Evaluation

This note documents the lightweight retrieval evaluation loop built into the pipeline CLI.

## Command

Run:

```bash
uv run python src/rag_pipeline.py --evaluate --skip-queries
```

This:
- loads and chunks the corpus
- rebuilds the vector store
- runs a fixed set of sample Norwegian questions
- prints a compact pass/fail report for top-1 retrieval

## Current Evaluation Cases

The current evaluation covers one representative question per client:
- Fjordmat tone of voice
- Spareklar Google Ads ROAS
- Nordvik influencer collaboration
- Skytjenester customer case results

Each case checks:
- expected client
- expected document type
- expected source file

## Current Success Rule

A case is considered `PASS` when the top-1 retrieved source matches the expected source file.

The report also shows whether:
- client metadata matched
- document type metadata matched
- source path matched

## Current Result

On the current dataset and ranking setup:
- top-1 source matches: `4/4`

## Why This Matters

This gives the project a simple repeatable quality check without adding testing framework complexity.

It is useful for:
- validating retrieval changes before merging
- spotting regressions quickly
- supporting the hiring-demo story with concrete evaluation evidence
