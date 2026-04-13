# Metadata Audit

This note documents the normalized metadata fields used by the retrieval pipeline.

## Command

Run the metadata audit with:

```bash
uv run python src/rag_pipeline.py --summary --metadata-audit --skip-queries
```

This:
- loads the corpus
- chunks the documents
- prints the corpus summary
- prints a metadata completeness audit
- rebuilds the local vector store

## Normalized Document Metadata

Each source document now carries:
- `client`
- `filename`
- `document_title`
- `document_type`
- `source`
- `source_path`

## Normalized Chunk Metadata

Each chunk carries the same source fields plus:
- `chunk_index`
- `chunk_id`

Example chunk identifier:

```text
skytjenester/seo-rapport-mars-2025.md#chunk-000
```

## Current Audit Result

On the current dataset:
- all required document fields are present on all 16 documents
- all required chunk fields are present on all 99 chunks
- document types are normalized into a small, explainable set

Current document type distribution:
- `brand_guidelines`: 4
- `campaign_brief`: 4
- `campaign_report`: 3
- `customer_case`: 1
- `influencer_strategy`: 1
- `meeting_notes`: 1
- `seo_report`: 1
- `social_media_strategy`: 1

## Why This Matters

This metadata shape supports the project priorities:
- stronger source attribution in retrieval results
- cleaner filtering in the future Streamlit UI
- easier debugging when retrieval quality needs improvement
- a more explainable pipeline for a hiring-demo setting
