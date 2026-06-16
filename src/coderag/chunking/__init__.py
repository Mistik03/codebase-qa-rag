"""Chunking strategies — the central variable studied in this thesis.

A *chunk* is the unit of retrieval: the codebase is split into chunks, each chunk
is embedded and indexed, and retrieval returns the chunks most relevant to a
question. How the code is split strongly affects whether the right context
reaches the model, so three strategies are implemented behind one interface and
compared head-to-head.
"""
from .base import Chunk, Chunker, estimate_tokens
from .naive import NaiveChunker
from .component import ComponentChunker
from .ast_chunker import AstChunker


def get_chunker(strategy: str, config: dict) -> Chunker:
    """Factory: build a chunker from its strategy name and the global config."""
    ch = config["chunking"]
    if strategy == "naive":
        return NaiveChunker(ch["window_tokens"], ch["overlap_tokens"])
    if strategy == "component":
        return ComponentChunker(ch["max_chunk_tokens"], ch["window_tokens"], ch["overlap_tokens"])
    if strategy == "ast":
        return AstChunker(ch["max_chunk_tokens"], ch["window_tokens"], ch["overlap_tokens"])
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


STRATEGIES = ["naive", "component", "ast"]

__all__ = ["Chunk", "Chunker", "estimate_tokens", "NaiveChunker",
           "ComponentChunker", "AstChunker", "get_chunker", "STRATEGIES"]
