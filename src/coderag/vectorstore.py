"""Persistent vector store backed by ChromaDB.

One collection per (codebase, chunking strategy) keeps experiments isolated and
re-runnable. Embeddings are supplied explicitly — Chroma is never asked to embed
anything itself, so no model is downloaded and the daemon stays fully local.
Cosine distance matches normalised text embeddings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chromadb


@dataclass
class Hit:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


def _safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return (name + "_idx")[:60]


class VectorStore:
    def __init__(self, persist_dir: str, collection: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.name = _safe_name(collection)
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        # Chroma caps batch size; add in slices to stay well under it.
        step = 256
        for i in range(0, len(ids), step):
            self.collection.add(
                ids=ids[i:i + step],
                embeddings=embeddings[i:i + step],
                documents=documents[i:i + step],
                metadatas=metadatas[i:i + step],
            )

    def query(self, embedding: list[float], n_results: int) -> list[Hit]:
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(Hit(id=cid, document=doc, metadata=meta, distance=float(dist)))
        return hits

    def count(self) -> int:
        return self.collection.count()
