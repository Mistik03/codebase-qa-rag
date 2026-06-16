"""FastAPI backend exposing the local codebase-QA assistant.

Endpoints
  GET  /options        available codebases / strategies / methods + defaults
  POST /query          {question, codebase, strategy, method, k?} -> answer + citations
  GET  /               the demo web page

Pipelines are cached per (codebase, strategy) and reuse the persisted ChromaDB
index, so a query only pays for retrieval + generation, not re-embedding.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coderag.chunking import STRATEGIES
from coderag.config import CONFIG
from coderag.metrics import ranked_files
from coderag.pipeline import RagPipeline
from coderag.retrieval import METHODS

app = FastAPI(title="codebase-qa-rag", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"
_pipelines: dict[tuple[str, str], RagPipeline] = {}


def get_pipeline(codebase: str, strategy: str, method: str) -> RagPipeline:
    key = (codebase, strategy)
    if key not in _pipelines:
        # reset=False reuses the persisted Chroma collection built by run_grid.
        _pipelines[key] = RagPipeline(codebase, strategy, method).build_index(reset=False)
    pipe = _pipelines[key]
    pipe.set_method(method)
    return pipe


class QueryRequest(BaseModel):
    question: str
    codebase: str = "microblog"
    strategy: str = "ast"
    method: str = "hybrid"
    k: int | None = None


@app.get("/options")
def options() -> dict:
    return {
        "codebases": list(CONFIG["codebases"]),
        "strategies": STRATEGIES,
        "methods": METHODS,
        "defaults": {"codebase": "microblog", "strategy": "ast", "method": "hybrid"},
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if req.codebase not in CONFIG["codebases"]:
        return {"error": f"unknown codebase '{req.codebase}'"}
    if req.strategy not in STRATEGIES or req.method not in METHODS:
        return {"error": "unknown strategy or method"}

    pipe = get_pipeline(req.codebase, req.strategy, req.method)
    res = pipe.query(req.question, req.k)
    return {
        "answer": res.answer,
        "error": res.error,
        "cited_files": res.cited_files,
        "retrieve_ms": round(res.retrieve_ms, 1),
        "generate_ms": round(res.generate_ms, 1),
        "config": {"codebase": req.codebase, "strategy": req.strategy,
                   "method": req.method, "model": CONFIG["ollama"]["gen_model"]},
        "retrieved": [
            {
                "rel_path": rc.rel_path,
                "start_line": rc.metadata.get("start_line"),
                "end_line": rc.metadata.get("end_line"),
                "score": round(rc.score, 4),
            }
            for rc in res.retrieved
        ],
        "retrieved_files": ranked_files(res.retrieved),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
