"""Dense retrieval: embed the query and find the nearest chunks by cosine."""
from __future__ import annotations

from ..embeddings import OllamaEmbedder
from ..vectorstore import VectorStore
from .base import RetrievedChunk


class DenseRetriever:
    name = "dense"

    def __init__(self, store: VectorStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        emb = self.embedder.embed_query(query)
        hits = self.store.query(emb, k)
        out: list[RetrievedChunk] = []
        for rank, h in enumerate(hits):
            out.append(RetrievedChunk(
                id=h.id,
                text=h.document,
                rel_path=h.metadata.get("rel_path", ""),
                files=RetrievedChunk.files_from_meta(h.metadata),
                score=1.0 - h.distance,   # cosine distance -> similarity
                rank=rank,
                metadata=h.metadata,
            ))
        return out
