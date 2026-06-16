"""Shared retrieval record."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    id: str
    text: str
    rel_path: str
    files: list[str]          # every source file this chunk covers
    score: float              # method-specific (similarity, BM25, or fused score)
    rank: int                 # 0-based rank within this retriever's result list
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def files_from_meta(meta: dict[str, Any]) -> list[str]:
        raw = meta.get("files") or meta.get("rel_path", "")
        return [f for f in str(raw).split(",") if f]
