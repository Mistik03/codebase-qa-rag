"""Hybrid retrieval: fuse dense and BM25 rankings with Reciprocal Rank Fusion.

RRF combines rankings without needing to calibrate the very different score
scales of cosine similarity and BM25. Each chunk's fused score is the sum over
both lists of ``1 / (rrf_k + rank)``; chunks ranked highly by either method
float to the top, and chunks ranked highly by both win.
"""
from __future__ import annotations

from .base import RetrievedChunk
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 rrf_k: int = 60, candidate_k: int = 20):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        dense_hits = self.dense.retrieve(query, self.candidate_k)
        bm25_hits = self.bm25.retrieve(query, self.candidate_k)

        fused: dict[str, float] = {}
        by_id: dict[str, RetrievedChunk] = {}
        for hits in (dense_hits, bm25_hits):
            for rc in hits:
                fused[rc.id] = fused.get(rc.id, 0.0) + 1.0 / (self.rrf_k + rc.rank)
                by_id.setdefault(rc.id, rc)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(ranked):
            rc = by_id[cid]
            out.append(RetrievedChunk(
                id=rc.id, text=rc.text, rel_path=rc.rel_path, files=rc.files,
                score=score, rank=rank, metadata=rc.metadata,
            ))
        return out
