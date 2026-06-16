"""Codebase loading: walk a project directory and read its source files.

The loader is deliberately language-agnostic. It returns a flat list of
``FileDoc`` objects that every chunking strategy consumes. Directory and glob
exclusions keep build artefacts, dependencies and generated code out of the
index so that retrieval operates on authored source only.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from .config import codebase_config, resolve_path

# Extension -> coarse language label used by chunkers and the AST grammars.
LANG_BY_EXT = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".py": "python",
}


@dataclass
class FileDoc:
    """A single source file loaded from a codebase."""

    path: str          # absolute path on disk
    rel_path: str      # path relative to the codebase root (posix style)
    text: str
    language: str
    ext: str
    n_lines: int


def _is_excluded(rel_parts: tuple[str, ...], name: str, rel_path: str,
                 exclude_dirs: set[str], exclude_glob: list[str]) -> bool:
    if exclude_dirs.intersection(rel_parts):
        return True
    for pattern in exclude_glob:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def load_codebase(root: str | Path, include_ext: list[str],
                  exclude_dirs: list[str] | None = None,
                  exclude_glob: list[str] | None = None) -> list[FileDoc]:
    """Load all in-scope source files from ``root``."""
    root_path = resolve_path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Codebase root does not exist: {root_path}")

    include = {e.lower() for e in include_ext}
    excl_dirs = set(exclude_dirs or [])
    excl_glob = list(exclude_glob or [])

    docs: list[FileDoc] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in include:
            continue
        rel = path.relative_to(root_path)
        rel_posix = rel.as_posix()
        if _is_excluded(rel.parts, path.name, rel_posix, excl_dirs, excl_glob):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        docs.append(FileDoc(
            path=str(path),
            rel_path=rel_posix,
            text=text,
            language=LANG_BY_EXT.get(path.suffix.lower(), "text"),
            ext=path.suffix.lower(),
            n_lines=text.count("\n") + 1,
        ))
    return sorted(docs, key=lambda d: d.rel_path)


def load_named_codebase(name: str) -> list[FileDoc]:
    """Load a codebase by its key in ``config.yaml`` (e.g. 'formfit')."""
    cfg = codebase_config(name)
    return load_codebase(
        root=cfg["root"],
        include_ext=cfg["include_ext"],
        exclude_dirs=cfg.get("exclude_dirs"),
        exclude_glob=cfg.get("exclude_glob"),
    )
