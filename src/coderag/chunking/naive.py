"""Naive chunking: fixed-size token windows with overlap, ignoring code structure.

This is the baseline. It treats every file as a flat stream of lines and cuts it
into overlapping windows of roughly ``window_tokens`` tokens. It is fast and
language-agnostic, but it routinely splits functions, classes and Angular
templates across chunk boundaries — which is exactly the weakness the
structure-aware strategies aim to fix.
"""
from __future__ import annotations

from ..loaders import FileDoc
from .base import CHARS_PER_TOKEN, Chunk, Chunker, line_windows, make_id


class NaiveChunker(Chunker):
    name = "naive"

    def __init__(self, window_tokens: int, overlap_tokens: int):
        self.window_chars = window_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    def chunk(self, files: list[FileDoc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in files:
            lines = doc.text.splitlines()
            for idx, (start, end, text) in enumerate(
                line_windows(lines, self.window_chars, self.overlap_chars)
            ):
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    id=make_id(self.name, doc.rel_path, idx),
                    text=text,
                    file_path=doc.path,
                    rel_path=doc.rel_path,
                    start_line=start,
                    end_line=end,
                    language=doc.language,
                    strategy=self.name,
                ))
        return chunks
