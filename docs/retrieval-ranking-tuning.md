# Retrieval Ranking Tuning

This note documents the first targeted retrieval-quality improvement.

## Problem

The baseline retrieval pipeline was usable, but it showed two recurring issues on the sample Norwegian questions:
- explicit client mentions did not reliably suppress cross-client noise
- some questions matched the right client but the wrong document type first

Examples from the earlier baseline:
- "Hva er Fjordmats tone of voice?" returned meeting notes above brand guidelines
- "Hva var ROAS for Spareklars Google Ads i Q4 2024?" included a Nordvik result in the top 2
- "Hvilke influencere samarbeider Nordvik med?" included a Fjordmat result in the top 2

## Change

The query path now applies lightweight query interpretation before returning results:
- infer `client` when the question names a client explicitly
- infer `document_type` from a few high-signal Norwegian phrases
- fetch a slightly larger candidate set
- rerank candidates so metadata matches are promoted

This stays simple and explainable:
- no provider-specific logic
- no external reranker
- no major architecture change

## Current Behavior On The Four Sample Questions

After the change:
- Fjordmat tone-of-voice questions promote `brand_guidelines`
- Spareklar ROAS questions stay on the Spareklar campaign report
- Nordvik influencer questions stay on the Nordvik influencer strategy
- Skytjenester customer-case questions promote `customer_case`

## Verification

Run:

```bash
uv run python src/rag_pipeline.py
```

The expected top results are now:
- Fjordmat question: `fjordmat/merkevareretningslinjer.md`
- Spareklar question: `spareklar/kampanjerapport-google-ads-q4-2024.md`
- Nordvik question: `nordvik/influencer-strategi.md`
- Skytjenester question: `skytjenester/kundecase-logistikkpartner.md`

## Why This Is A Good MVP Tradeoff

This improves retrieval quality using project metadata that already exists, which fits the demo well:
- practical
- easy to explain in an interview
- low complexity
- directly tied to source-grounded answers
