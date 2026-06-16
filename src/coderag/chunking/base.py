"""Shared chunking primitives: the ``Chunk`` record, the ``Chunker`` interface,
a consistent token estimator, and a robust line-window splitter reused by every
strategy as a fallback for oversized units.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..loaders import FileDoc

# A deliberately simple, consistent approximation. Using one ratio everywhere
# keeps the three strategies comparable without pulling in a tokenizer.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    """One unit of retrieval, with provenance back to its source lines."""

    id: str
    text: str
    file_path: str
    rel_path: str
    start_line: int
    end_line: int
    language: str
    strategy: str
    component: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Flatten to Chroma-safe primitive metadata.

        ``files`` is a comma-separated list of every source file the chunk
        covers (more than one for grouped component chunks); metrics use it to
        credit retrieval at file granularity.
        """
        meta = {
            "file_path": self.file_path,
            "rel_path": self.rel_path,
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "language": self.language,
            "strategy": self.strategy,
            "component": self.component or "",
            "files": self.metadata.get("files", self.rel_path),
        }
        # Allow strategies to attach extra primitive metadata.
        for key, value in self.metadata.items():
            if key not in meta and isinstance(value, (str, int, float, bool)):
                meta[key] = value
        return meta

    def covered_files(self) -> list[str]:
        raw = self.metadata.get("files", self.rel_path)
        return [f for f in raw.split(",") if f]


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, files: list[FileDoc]) -> list[Chunk]:
        """Split a loaded codebase into retrieval chunks."""


def make_id(strategy: str, rel_path: str, index: int) -> str:
    return f"{strategy}|{rel_path}|{index}"


def line_windows(lines: list[str], window_chars: int, overlap_chars: int):
    """Yield ``(start_line, end_line, text)`` windows over ``lines``.

    Line numbers are 1-based and inclusive. Guarantees forward progress even
    when a single line is larger than the window, so it can never loop forever.
    """
    n = len(lines)
    if n == 0:
        return
    i = 0
    while i < n:
        start = i
        size = 0
        while i < n and (size < window_chars or i == start):
            size += len(lines[i]) + 1
            i += 1
        end = i  # exclusive
        yield start + 1, end, "\n".join(lines[start:end])
        if i >= n:
            break
        # Step back to create overlap, but always net-advance by >= 1 line.
        if overlap_chars > 0:
            back, acc, j = 0, 0, end - 1
            while j > start and acc < overlap_chars:
                acc += len(lines[j]) + 1
                back += 1
                j -= 1
            back = min(back, (end - start) - 1)
            i = end - back
