"""Command-line entry point for a single question against a codebase.

Example:
    python -m experiments.ask --codebase microblog --strategy ast --method hybrid \
        "How does user authentication work?"
"""
from __future__ import annotations

import argparse

from coderag.chunking import STRATEGIES
from coderag.config import CONFIG
from coderag.pipeline import RagPipeline
from coderag.retrieval import METHODS


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a local RAG assistant about a codebase.")
    ap.add_argument("question")
    ap.add_argument("--codebase", default="microblog", choices=list(CONFIG["codebases"]))
    ap.add_argument("--strategy", default="ast", choices=STRATEGIES)
    ap.add_argument("--method", default="hybrid", choices=METHODS)
    ap.add_argument("-k", type=int, default=None, help="number of chunks to retrieve")
    ap.add_argument("--reindex", action="store_true", help="rebuild the vector index")
    args = ap.parse_args()

    print(f"Indexing {args.codebase} with strategy={args.strategy} ...")
    pipe = RagPipeline(args.codebase, args.strategy, args.method).build_index(reset=args.reindex)
    print(f"Indexed {len(pipe.chunks)} chunks. Asking ...\n")

    res = pipe.query(args.question, args.k)
    if res.error:
        print("ERROR:", res.error)
        return
    print("=" * 70)
    print(res.answer)
    print("=" * 70)
    print("\nCited files:", ", ".join(res.cited_files) or "(none)")
    print("Retrieved (top):")
    for rc in res.retrieved:
        loc = f"{rc.metadata.get('start_line', '?')}-{rc.metadata.get('end_line', '?')}"
        print(f"  [{rc.rank}] {rc.rel_path}:{loc}  score={rc.score:.3f}")
    print(f"\nretrieve={res.retrieve_ms:.0f} ms  generate={res.generate_ms:.0f} ms")


if __name__ == "__main__":
    main()
