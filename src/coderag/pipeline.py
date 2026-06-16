"""End-to-end RAG pipeline: index a codebase, then answer questions about it.

A pipeline is parameterised by the three experimental axes (codebase, chunking
strategy, retrieval method). The dense index is built once per
(codebase, strategy); the retrieval method can then be switched without
re-embedding, which is what lets the experiment grid reuse work efficiently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk, get_chunker
from .config import CONFIG, resolve_path
from .embeddings import embedder_from_config
from .llm import generator_from_config
from .loaders import load_named_codebase
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RetrievedChunk
from .vectorstore import VectorStore


@dataclass
class QueryResult:
    question: str
    answer: str
    cited_files: list[str]
    retrieved: list[RetrievedChunk]
    retrieve_ms: float
    generate_ms: float
    error: str | None = None
    meta: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(self, codebase: str, strategy: str, method: str, config: dict = CONFIG):
        self.codebase = codebase
        self.strategy = strategy
        self.method = method
        self.config = config
        r = config["retrieval"]
        self.top_k = r["top_k"]
        self.candidate_k = r["candidate_k"]
        self.rrf_k = r["rrf_k"]

        self.embedder = embedder_from_config(config)
        self.generator = generator_from_config(config)
        self.chunks: list[Chunk] = []
        self.store: VectorStore | None = None
        self.dense: DenseRetriever | None = None
        self.bm25: BM25Retriever | None = None
        self.retriever = None

    def build_index(self, reset: bool = True) -> "RagPipeline":
        files = load_named_codebase(self.codebase)
        self.chunks = get_chunker(self.strategy, self.config).chunk(files)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced for {self.codebase}/{self.strategy}")

        persist = str(resolve_path(self.config["storage"]["chroma_dir"]))
        self.store = VectorStore(persist, f"{self.codebase}_{self.strategy}")
        if reset:
            self.store.reset()
        if self.store.count() == 0:
            texts = [c.text for c in self.chunks]
            self.store.add(
                ids=[c.id for c in self.chunks],
                embeddings=self.embedder.embed_documents(texts),
                documents=texts,
                metadatas=[c.to_metadata() for c in self.chunks],
            )

        chunk_dicts = [{
            "id": c.id, "text": c.text, "rel_path": c.rel_path,
            "files": c.covered_files(), "metadata": c.to_metadata(),
        } for c in self.chunks]
        self.dense = DenseRetriever(self.store, self.embedder)
        self.bm25 = BM25Retriever(chunk_dicts)
        self.set_method(self.method)
        return self

    def set_method(self, method: str) -> "RagPipeline":
        self.method = method
        if method == "dense":
            self.retriever = self.dense
        elif method == "bm25":
            self.retriever = self.bm25
        elif method == "hybrid":
            self.retriever = HybridRetriever(self.dense, self.bm25, self.rrf_k, self.candidate_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")
        return self

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, k or self.top_k)

    def query(self, question: str, k: int | None = None) -> QueryResult:
        k = k or self.top_k
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(question, k)
        t1 = time.perf_counter()
        ans = self.generator.answer(question, retrieved)
        t2 = time.perf_counter()
        return QueryResult(
            question=question,
            answer=ans.answer,
            cited_files=ans.cited_files,
            retrieved=retrieved,
            retrieve_ms=(t1 - t0) * 1000,
            generate_ms=(t2 - t1) * 1000,
            error=ans.error,
            meta={"strategy": self.strategy, "method": self.method,
                  "codebase": self.codebase, "n_chunks": len(self.chunks)},
        )
