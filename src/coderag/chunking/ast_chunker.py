"""AST-aware chunking: split at syntactic boundaries using tree-sitter.

Instead of cutting at arbitrary line counts, this strategy parses each file and
cuts at top-level definition boundaries — functions and classes — so that a
chunk is a whole, self-contained unit of code. Decorators are absorbed
automatically (Python ``decorated_definition`` and TypeScript ``export_statement``
include them), and code before the first definition (imports, module constants)
forms a preamble chunk so the whole file is still covered.

Files without a supported grammar (HTML, CSS, SCSS) fall back to the shared
line-window splitter, as does any file that fails to parse.

The tree-sitter binding shipped by ``tree-sitter-language-pack`` exposes node
accessors as *methods* (``root_node()``, ``child_count()``, ``kind()``,
``start_position()``), which differs from the classic py-tree-sitter API; the
small ``_call`` shim tolerates either form.
"""
from __future__ import annotations

from ..loaders import FileDoc
from .base import CHARS_PER_TOKEN, Chunk, Chunker, line_windows, make_id

_GRAMMAR_BY_EXT = {".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript"}

_TARGETS = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "typescript": {
        "function_declaration", "generator_function_declaration", "class_declaration",
        "abstract_class_declaration", "interface_declaration", "enum_declaration",
        "type_alias_declaration", "lexical_declaration", "export_statement",
    },
}
_TARGETS["tsx"] = _TARGETS["typescript"]
_TARGETS["javascript"] = {
    "function_declaration", "generator_function_declaration", "class_declaration",
    "lexical_declaration", "export_statement",
}


def _call(attr):
    """Return ``attr()`` if it is callable, else ``attr`` (method/property shim)."""
    return attr() if callable(attr) else attr


class AstChunker(Chunker):
    name = "ast"

    def __init__(self, max_chunk_tokens: int, window_tokens: int, overlap_tokens: int):
        self.max_chars = max_chunk_tokens * CHARS_PER_TOKEN
        self.window_chars = window_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN
        self._parsers: dict[str, object] = {}
        self._pack_ok: bool | None = None

    def _get_parser(self, grammar: str | None):
        if not grammar or self._pack_ok is False:
            return None
        if grammar in self._parsers:
            return self._parsers[grammar]
        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(grammar)
            self._pack_ok = True
            self._parsers[grammar] = parser
            return parser
        except Exception:
            if grammar == "python":  # core grammar missing -> disable entirely
                self._pack_ok = False
            return None

    def chunk(self, files: list[FileDoc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in files:
            parser = self._get_parser(_GRAMMAR_BY_EXT.get(doc.ext))
            if parser is None:
                chunks.extend(self._fallback(doc))
                continue
            try:
                chunks.extend(self._ast_chunk(doc, parser, _GRAMMAR_BY_EXT[doc.ext]))
            except Exception:
                chunks.extend(self._fallback(doc))
        return chunks

    def _ast_chunk(self, doc: FileDoc, parser, grammar: str) -> list[Chunk]:
        targets = _TARGETS.get(grammar, set())
        try:
            tree = parser.parse(doc.text)
        except TypeError:
            tree = parser.parse(doc.text.encode("utf-8"))
        root = _call(tree.root_node)
        lines = doc.text.split("\n")

        cuts = {0}
        for i in range(_call(root.child_count)):
            child = root.child(i)
            if _call(child.kind) in targets:
                row = _call(child.start_position).row
                if 0 <= row < len(lines):
                    cuts.add(row)

        ordered = sorted(cuts) + [len(lines)]
        out: list[Chunk] = []
        idx = 0
        for a, b in zip(ordered, ordered[1:]):
            segment = "\n".join(lines[a:b])
            if not segment.strip():
                continue
            if len(segment) <= self.max_chars:
                out.append(self._mk(doc, idx, segment, a + 1, b))
                idx += 1
            else:
                for ws, we, wtext in line_windows(lines[a:b], self.window_chars, self.overlap_chars):
                    if wtext.strip():
                        out.append(self._mk(doc, idx, wtext, a + ws, a + we))
                        idx += 1
        return out or self._fallback(doc)

    def _mk(self, doc: FileDoc, idx: int, text: str, start: int, end: int) -> Chunk:
        return Chunk(
            id=make_id(self.name, doc.rel_path, idx),
            text=text,
            file_path=doc.path,
            rel_path=doc.rel_path,
            start_line=start,
            end_line=end,
            language=doc.language,
            strategy=self.name,
        )

    def _fallback(self, doc: FileDoc) -> list[Chunk]:
        out: list[Chunk] = []
        for idx, (start, end, text) in enumerate(
            line_windows(doc.text.splitlines(), self.window_chars, self.overlap_chars)
        ):
            if text.strip():
                out.append(self._mk(doc, idx, text, start, end))
        return out
