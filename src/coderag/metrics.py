"""File-level retrieval metrics.

Gold labels mark which *files* should be retrieved for each question, so metrics
are computed at file granularity. Retrieved chunks are flattened into an ordered
list of distinct files (a grouped component chunk contributes all its files at
its rank), then scored with precision@k, recall@k and reciprocal rank.
"""
from __future__ import annotations

from .retrieval.base import RetrievedChunk


def ranked_files(retrieved: list[RetrievedChunk]) -> list[str]:
    """Distinct files in retrieval order."""
    seen: list[str] = []
    for rc in retrieved:
        for f in rc.files:
            if f and f not in seen:
                seen.append(f)
    return seen


def precision_at_k(retrieved_files: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_files[:k]
    if not top:
        return 0.0
    hits = sum(1 for f in top if f in gold)
    return hits / len(top)


def recall_at_k(retrieved_files: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved_files[:k])
    return len(top & gold) / len(gold)


def reciprocal_rank(retrieved_files: list[str], gold: set[str]) -> float:
    for i, f in enumerate(retrieved_files, 1):
        if f in gold:
            return 1.0 / i
    return 0.0


def score_retrieval(retrieved: list[RetrievedChunk], gold: list[str],
                    k: int) -> dict[str, float]:
    files = ranked_files(retrieved)
    gold_set = set(gold)
    return {
        f"precision@{k}": precision_at_k(files, gold_set, k),
        f"recall@{k}": recall_at_k(files, gold_set, k),
        "mrr": reciprocal_rank(files, gold_set),
        "n_retrieved_files": float(len(files)),
    }
