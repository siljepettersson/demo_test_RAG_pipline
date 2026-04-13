# Pipeline Readability

This note documents a small refactor that improves how the pipeline can be explained in a demo or interview.

## Goal

The goal of this step is not to change retrieval behavior.

The goal is to make the code easier to present by:
- reducing repeated inline logic
- grouping related responsibilities more clearly
- making the CLI flow easier to follow

## What Changed

The current cleanup keeps behavior the same while making the code easier to read:
- repeated metadata field lists are defined once as constants
- retrieval search kwargs are built through a small helper
- the CLI path now runs through a dedicated `run_cli()` function

This makes the main flow easier to describe as:
1. inspect corpus
2. optionally print audits
3. build vector store
4. optionally run evaluation
5. optionally print example query results

## Why This Matters

For this repo, explainability matters almost as much as functionality.

A hiring manager should be able to understand the pipeline quickly:
- how documents are loaded
- how chunks get metadata
- how retrieval is improved with lightweight reranking
- how quality is evaluated

This refactor supports that without adding unnecessary abstraction.
