"""FastAPI service exposing the assistant over HTTP.

New code -- there is no notebook equivalent (the notebook is a prototyping
environment, not a server). Builds/loads the Qdrant index and NetworkX graph
ONCE at process startup (never per-request), reusing
`src/ingestion/index_build.py` and `src/graphrag/graph_build.py` directly;
`load_or_build_graph()` loads a persisted graph from disk instead of
rebuilding it via LLM calls if one already exists.

The OpenAI key never reaches client-side code -- `static/index.html` only
ever calls this service's own `/ask` endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import Config
from src.data.loaders import load_corpus, load_gap_taxonomy
from src.graphrag.community import detect_communities, summarize_community
from src.graphrag.graph_build import load_or_build_graph
from src.ingestion.index_build import build_index
from src.llm.backend import OpenAIBackend
from src.orchestration.graph import ask as ask_assistant
from src.orchestration.graph import init_assistant_graph
from src.orchestration.nodes import Deps
from src.retrieval.local_query import LocalQueryEngine

logger = logging.getLogger("neurosurgery_graphrag_assistant.api")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

_config = Config.from_env()

# Demo-grade rate limiting: a real, billed OpenAI call happens on every /ask
# request with no caching layer in this version -- see docs/ARCHITECTURE.md's
# "Rate limiting" section for exactly why this is documented as demo-grade
# (in-memory, per-process, resets on restart) rather than production-grade.
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Builds/loads the Qdrant index and NetworkX graph ONCE at process startup
    (never per-request)."""
    deps = build_startup_deps(_config)
    init_assistant_graph(deps)
    logger.info("Startup complete: index and graph built/loaded, assistant graph initialized.")
    yield


app = FastAPI(title="Neurosurgery-AI GraphRAG Research Assistant", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allow the deployed frontend origin only, read from ALLOWED_ORIGIN.
# "*" is the default, and is intended for local/dev only -- never for a
# documented production config (see .env.example / render.yaml / fly.toml).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_config.ALLOWED_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A research question about the corpus.")


class AskResponse(BaseModel):
    answer: str
    retrieved_paper_ids: list[str]
    intent: str


def build_startup_deps(config: Config) -> Deps:
    """Builds/loads the Qdrant index and NetworkX graph once, and bundles
    everything the orchestration graph needs. Kept as its own function so
    tests can mock this single call rather than every heavy dependency it
    wires up."""
    llm_backend = OpenAIBackend(config)
    corpus = load_corpus(mode=config.CORPUS_MODE)
    gap_taxonomy = load_gap_taxonomy().root

    index = build_index(corpus, config=config)
    local_query_engine = LocalQueryEngine(index, llm_backend)

    graph = load_or_build_graph(corpus, gap_taxonomy, llm_backend)
    communities = detect_communities(graph)
    community_summaries = [summarize_community(graph, c, gap_taxonomy, llm_backend) for c in communities]

    return Deps(
        llm_backend=llm_backend,
        local_query_engine=local_query_engine,
        community_summaries=community_summaries,
        corpus=corpus,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index_page() -> FileResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="static/index.html is missing from this deployment.")
    return FileResponse(INDEX_HTML_PATH)


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    try:
        result = ask_assistant(body.question)
    except Exception as e:
        # A real backend failure (e.g. a misconfigured/invalid OPENAI_API_KEY, which
        # is a genuine bug and deliberately NOT masked as a quota fallback -- see
        # src/llm/backend.py's is_quota_error()) must surface as a clear, documented
        # error, never a bare, unexplained 500.
        logger.exception("ask_assistant() failed while answering a question")
        raise HTTPException(
            status_code=502, detail=f"The assistant backend failed to answer this question: {e}"
        ) from e
    return AskResponse(
        answer=result.get("answer", ""),
        retrieved_paper_ids=result.get("retrieved_paper_ids", []),
        intent=result.get("intent", ""),
    )
