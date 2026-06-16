"""BM25 lexical retrieval over chunk text.

Code identifiers are split on both non-alphanumeric boundaries and camelCase /
PascalCase, and both the whole identifier and its sub-tokens are kept. This lets
a query like "user authentication" match ``authenticate_user`` and
``UserAuthService`` that a purely dense embedding can miss.
"""
from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from .base import RetrievedChunk

_SUBTOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize_code(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.split(r"[^A-Za-z0-9]+", text):
        if not raw:
            continue
        tokens.append(raw.lower())
        parts = _SUBTOKEN.findall(raw)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]]):
        # chunks: dicts with keys id, text, rel_path, files, metadata
        self.chunks = chunks
        corpus = [tokenize_code(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_code(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for rank, i in enumerate(order):
            c = self.chunks[i]
            out.append(RetrievedChunk(
                id=c["id"],
                text=c["text"],
                rel_path=c["rel_path"],
                files=c["files"],
                score=float(scores[i]),
                rank=rank,
                metadata=c["metadata"],
            ))
        return out
