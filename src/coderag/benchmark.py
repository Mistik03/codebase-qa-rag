"""Loading and validating the hand-labelled QA benchmark.

Each codebase has a ``benchmark/<codebase>.jsonl`` file whose lines are objects
with ``id``, ``question``, ``reference_answer``, ``gold_files`` (the files that
must be retrieved to answer) and ``tags``. ``gold_files`` are paths relative to
the codebase root, matching the ``rel_path`` produced by the loader.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import CONFIG, resolve_path
from .loaders import load_named_codebase


@dataclass
class BenchItem:
    id: str
    question: str
    reference_answer: str
    gold_files: list[str]
    codebase: str
    tags: list[str] = field(default_factory=list)


def benchmark_path(codebase: str):
    return resolve_path(CONFIG["paths"]["benchmark_dir"]) / f"{codebase}.jsonl"


def load_benchmark(codebase: str) -> list[BenchItem]:
    path = benchmark_path(codebase)
    items: list[BenchItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        items.append(BenchItem(
            id=d["id"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            gold_files=d["gold_files"],
            codebase=codebase,
            tags=d.get("tags", []),
        ))
    return items


def validate_benchmark(codebase: str) -> list[str]:
    """Return a list of problems (missing gold files / dup ids); empty == OK."""
    items = load_benchmark(codebase)
    present = {doc.rel_path for doc in load_named_codebase(codebase)}
    problems: list[str] = []
    seen_ids: set[str] = set()
    for it in items:
        if it.id in seen_ids:
            problems.append(f"duplicate id: {it.id}")
        seen_ids.add(it.id)
        if not it.gold_files:
            problems.append(f"{it.id}: no gold_files")
        for gf in it.gold_files:
            if gf not in present:
                problems.append(f"{it.id}: gold file not in index: {gf}")
    return problems
