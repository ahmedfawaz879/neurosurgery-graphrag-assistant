# Architecture

## Overview

This repo turns the `neurosurgery-graphrag-assistant` notebook's pipeline into a served
application: a FastAPI service wraps a LangGraph orchestration graph that routes each
question to one of two retrieval paths -- a vector-index local-query engine, or a
GraphRAG global-search path over community summaries of an LLM-extracted knowledge
graph -- and verifies attribution before returning an answer.

```
                        ┌─────────────────────┐
 14-paper corpus  ───▶  │  Ingestion           │
 (data/corpus/*.jsonl)  │  chunk + embed        │──▶ Qdrant (vector index)
                        └─────────────────────┘
                                  │
                        ┌─────────────────────┐
                        │  Triple extraction    │──▶ NetworkX graph (data/graph/*.graphml)
                        │  + community detect.  │       (Neo4j export available, unexercised)
                        └─────────────────────┘
                                  │
                        ┌─────────────────────┐
   question  ───────▶   │  LangGraph router     │
                        │  classify_intent       │
                        └──────────┬────────────┘
                          local    │    global
                     ┌─────────────┴─────────────┐
                     ▼                             ▼
         LocalQueryEngine                graphrag_global_search
         (Qdrant vector search)          (map-reduce over community
                     │                     summaries)
                     └─────────────┬─────────────┘
                                   ▼
                        verify_attribution_node
                         (revise once if UNATTRIBUTED)
                                   │
                                   ▼
                        FastAPI  POST /ask  ──▶  static/index.html
```

## Components

| Layer | Module | Notes |
|---|---|---|
| Config | `src/config.py` | Single `Config` dataclass, loaded from env vars. |
| LLM backend | `src/llm/backend.py` | `OpenAIBackend.generate()`; auto-falls back to a local Qwen model on an OpenAI quota/billing error. |
| Data | `src/data/` | Pydantic schemas + loaders for the corpus, QA set, gap taxonomy. |
| Ingestion | `src/ingestion/index_build.py` | Builds the Qdrant vector index; embeddings fall back to a local sentence-transformers model on an OpenAI embeddings quota error. |
| Retrieval | `src/retrieval/` | `LocalQueryEngine` (vector search) + Qdrant/Pinecone backend selection. |
| GraphRAG | `src/graphrag/` | Triple extraction, graph build/persistence, Louvain community detection + summarization, local (graph-traversal) and global (map-reduce) search, Neo4j export. |
| Orchestration | `src/orchestration/` | LangGraph `StateGraph`: intent classification → generation → attribution verification → (at most one) revision. |
| Evaluation | `src/eval/` | Citation-attribution accuracy, gap-resolution accuracy, global-coverage recall, false-attribution rate, bootstrap CIs, the harness runner. |
| Reporting | `src/reporting/` | Aggregation + bar-with-CI plots. |
| API | `src/api/main.py` | FastAPI service: `/ask`, `/health`, `/`. |
| UI | `static/index.html` | Single self-contained page, no build step, no external CDN dependency. |

## Design decisions worth calling out explicitly

- **NetworkX is the default graph backend.** `to_neo4j_cypher()` and `push_to_neo4j()`
  are real, complete code paths (`src/graphrag/neo4j_export.py`), but Neo4j is opt-in
  (`Config.GRAPH_BACKEND == "neo4j"`) and untested against a live instance in this
  repo -- NetworkX keeps the app runnable with zero external graph service.
- **Louvain is the default community-detection algorithm**, with an optional Leiden
  path behind extra, non-default dependencies (`src/graphrag/community.py`).
- **Gap-tag classification fails closed.** An LLM response with an invalid or missing
  `gap_code` is coerced to `None` (`src/graphrag/triple_extraction.py`), never silently
  accepted as a real taxonomy code.
- **Two independent local-search implementations exist on purpose.**
  `src/retrieval/local_query.py`'s embedding-based `LocalQueryEngine` is the primary
  vector-search path; `src/graphrag/local_search.py`'s `graphrag_local_search()` is a
  deliberately simple keyword-overlap graph-traversal alternative, not tuned to
  compete with it.
- **Dual OpenAI-quota fallback.** Chat completions (`OpenAIBackend.generate()`) and
  embeddings (`configure_embed_model()` in `src/ingestion/index_build.py`) are two
  *separate* OpenAI dependencies, each with its own independent quota/billing-error
  fallback to a local model -- discovered as two separate bugs during the notebook's
  authoring, and both preserved here as real, tested code paths.
- **`ALLOWED_ORIGIN` CORS is restricted in production.** The default is `*`, which is
  documented as local/dev-only; a deployed instance should set `ALLOWED_ORIGIN` to the
  exact frontend origin.

## Rate limiting

`POST /ask` is rate-limited via `slowapi` (`src/api/main.py`), currently `10/minute`
per client IP. **This is a demo-grade limiter, not a production-grade one**: it is
in-memory, scoped to a single process, and resets on every restart or deploy -- it
will not coordinate correctly across multiple replicas of the service. This app makes
a real, billed OpenAI API call on every `/ask` request with no caching layer in this
version, so a public deployment should also configure an OpenAI usage cap / budget
alert on the account, independent of this limiter (see the Deployment section below
and the README's Deploy section for the full warning).

## Deployment

See the **Deployment** section of the README for the one-command deploy paths
(`render.yaml` / `fly.toml`) and the cost/rate-limit warning that must be read before
deploying a public instance. Both configs and the Qdrant-hosting options they assume
are documented in detail below.

### Qdrant hosting for a cloud deployment

Two options, either is fine for a portfolio-scale deployment:

1. **Qdrant Cloud's free tier** (a managed cluster, external to whichever platform
   runs the API container) -- `QDRANT_URL` points at the cluster's HTTPS endpoint,
   `QDRANT_API_KEY` authenticates to it (see `src/retrieval/vector_store_backends.py`).
   Nothing to keep running yourself; this repo's **default** for both `render.yaml`
   and `fly.toml`, because it keeps each config a single deployable service --
   "a working config a visitor can actually click Deploy on" (per this section's own
   design goal) beats a multi-service Blueprint that's harder to verify sight-unseen.
2. **A second container on the same platform** (a separate Render private service,
   or a second Fly.io app in the same organization) running the official
   `qdrant/qdrant` image, with `QDRANT_URL` pointing at its internal address and no
   `QDRANT_API_KEY` needed. This is what `docker-compose.yml` does for local/
   self-hosted use, and is a reasonable alternative for a cloud deploy too -- it
   just means adding a second service block to `render.yaml` / a second `fly.toml`
   app yourself, following the same shape as the `qdrant` service in
   `docker-compose.yml`.

Switching between them is a `QDRANT_URL` (+ `QDRANT_API_KEY`) change, nothing else in
the application code.

### Cost and rate-limit warning

**This app makes a real, billed OpenAI API call on every `/ask` request, with no
caching layer in this version.** A portfolio visitor clicking around the public demo
generates real cost. Before sharing a live link publicly:

- Keep the demo-grade rate limiter (`10/minute` per IP) in place, or lower it further
  for a public link, **and/or**
- Configure an OpenAI usage cap / budget alert on the account running the deployment.

Do not deploy a version without a rate limit and call the demo "production-grade" --
it is not; see the "Rate limiting" section above for exactly why.
