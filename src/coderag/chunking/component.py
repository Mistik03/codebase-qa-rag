"""Component / module-aware chunking: keep structurally related code together.

The same idea — *group code that belongs together* — maps differently across
ecosystems, and that mapping is itself a finding of the thesis:

* **Angular** spreads one component across sibling files that share a name:
  ``footer.ts``, ``footer.html``, ``footer.css`` (modern Angular) or
  ``foo.component.{ts,html,css}`` (classic). Here a chunk is the whole component:
  logic, template and styles concatenated, so the model sees them as one unit.
* **Python / Flask** keeps a unit's logic in a single module, so the natural
  grouping is the whole file: a ``.py`` module (or a Jinja template) is one chunk.

Grouping is by ``(directory, stem)``: files in the same directory whose name
(minus the final extension) matches are treated as one component. Self-adapting
— Angular projects form multi-file groups while Python projects fall back to
single-file (module) chunks with no language flag required. Oversized units are
split with the shared line-window splitter while staying tagged to their unit.
"""
from __future__ import annotations

from pathlib import PurePosixPath

from ..loaders import FileDoc
from .base import CHARS_PER_TOKEN, Chunk, Chunker, line_windows, make_id

# Order files within a component group logically: logic, template, then styles.
_EXT_ORDER = {".ts": 0, ".tsx": 0, ".js": 1, ".html": 2, ".scss": 3, ".css": 4, ".py": 0}


def group_key(rel_path: str) -> str:
    """Shared key for files of the same component/module: '<dir>/<stem>'."""
    p = PurePosixPath(rel_path)
    parent = p.parent.as_posix()
    return f"{parent}/{p.stem}" if parent != "." else p.stem


class ComponentChunker(Chunker):
    name = "component"

    def __init__(self, max_chunk_tokens: int, window_tokens: int, overlap_tokens: int):
        self.max_chars = max_chunk_tokens * CHARS_PER_TOKEN
        self.window_chars = window_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    def chunk(self, files: list[FileDoc]) -> list[Chunk]:
        groups: dict[str, list[FileDoc]] = {}
        for doc in files:
            groups.setdefault(group_key(doc.rel_path), []).append(doc)

        chunks: list[Chunk] = []
        for key in sorted(groups):
            docs = groups[key]
            if len(docs) > 1:
                chunks.extend(self._emit_group(key, docs))
            else:
                chunks.extend(self._emit_single(docs[0]))
        return chunks

    # -- multi-file component: concatenate the sibling files into one unit -------
    def _emit_group(self, key: str, docs: list[FileDoc]) -> list[Chunk]:
        docs = sorted(docs, key=lambda d: _EXT_ORDER.get(d.ext, 9))
        primary = docs[0]
        files_csv = ",".join(d.rel_path for d in docs)
        combined = "\n\n".join(f"// ===== {d.rel_path} =====\n{d.text}" for d in docs)
        meta = {"files": files_csv}
        language = "angular" if any(d.ext in (".ts", ".tsx") for d in docs) else primary.language

        if len(combined) <= self.max_chars:
            return [self._mk(primary, key, 0, combined, 1, primary.n_lines,
                             component=key, language=language, meta=meta)]
        out: list[Chunk] = []
        for idx, (start, end, text) in enumerate(
            line_windows(combined.splitlines(), self.window_chars, self.overlap_chars)
        ):
            if text.strip():
                out.append(self._mk(primary, key, idx, text, start, end,
                                    component=key, language=language, meta=dict(meta)))
        return out

    # -- single file (Python module / standalone .ts): whole file is the unit ----
    def _emit_single(self, doc: FileDoc) -> list[Chunk]:
        if len(doc.text) <= self.max_chars:
            return [self._mk(doc, doc.rel_path, 0, doc.text, 1, doc.n_lines,
                             component=doc.rel_path, language=doc.language, meta={})]
        out: list[Chunk] = []
        for idx, (start, end, text) in enumerate(
            line_windows(doc.text.splitlines(), self.window_chars, self.overlap_chars)
        ):
            if text.strip():
                out.append(self._mk(doc, doc.rel_path, idx, text, start, end,
                                    component=doc.rel_path, language=doc.language, meta={}))
        return out

    def _mk(self, primary: FileDoc, key: str, idx: int, text: str, start: int,
            end: int, component: str, language: str, meta: dict) -> Chunk:
        return Chunk(
            id=make_id(self.name, key, idx),
            text=text,
            file_path=primary.path,
            rel_path=primary.rel_path,
            start_line=start,
            end_line=end,
            language=language,
            strategy=self.name,
            component=component,
            metadata=meta,
        )
