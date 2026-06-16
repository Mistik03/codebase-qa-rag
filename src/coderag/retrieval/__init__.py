"""Retrieval methods: dense (vector) and hybrid (dense + BM25 via RRF)."""
from .base import RetrievedChunk
from .dense import DenseRetriever
from .bm25 import BM25Retriever, tokenize_code
from .hybrid import HybridRetriever

METHODS = ["dense", "hybrid"]

__all__ = ["RetrievedChunk", "DenseRetriever", "BM25Retriever",
           "HybridRetriever", "tokenize_code", "METHODS"]
