"""Run the full experiment grid: codebase x chunking x retrieval.

Scope (leanest variant, per the thesis time budget): the generation model is
fixed to Qwen2.5-Coder 3B and the grid is

    {microblog, formfit} x {naive, component, ast} x {dense, hybrid}

evaluated over the hand-labelled benchmark (~20 questions). For each
(config, question) we record retrieval metrics (precision/recall/MRR at file
level), the generated answer with its latency, and an LLM-judge quality score.

To avoid reloading models on a 6 GB GPU, the run is two-phase:
  Phase A — retrieve + generate for everything (Qwen + nomic resident);
  Phase B — judge every collected answer (Llama-3.1-8B resident).

Outputs (experiments/results/):
  grid_runs.jsonl   full per-question records, incl. answers and citations
  grid_results.csv  one tabular row per (config, question)
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from coderag.benchmark import load_benchmark
from coderag.chunking import STRATEGIES
from coderag.config import CONFIG, resolve_path
from coderag.judge import judge_from_config
from coderag.metrics import ranked_files, score_retrieval
from coderag.pipeline import RagPipeline
from coderag.retrieval import METHODS

CODEBASES = ["microblog", "formfit"]
TOP_K = CONFIG["retrieval"]["top_k"]


def phase_a_generate() -> list[dict]:
    runs: list[dict] = []
    for codebase in CODEBASES:
        items = load_benchmark(codebase)
        for strategy in STRATEGIES:
            print(f"[index] {codebase}/{strategy} ...", flush=True)
            pipe = RagPipeline(codebase, strategy, "dense").build_index(reset=True)
            print(f"[index] {codebase}/{strategy}: {len(pipe.chunks)} chunks", flush=True)
            for method in METHODS:
                pipe.set_method(method)
                for it in items:
                    try:
                        res = pipe.query(it.question, TOP_K)
                        m = score_retrieval(res.retrieved, it.gold_files, TOP_K)
                        runs.append({
                            "codebase": codebase, "strategy": strategy, "method": method,
                            "qid": it.id, "tags": it.tags, "gold_files": it.gold_files,
                            "question": it.question, "reference": it.reference_answer,
                            "answer": res.answer, "cited_files": res.cited_files,
                            "retrieved_files": ranked_files(res.retrieved)[:TOP_K],
                            "precision": m[f"precision@{TOP_K}"],
                            "recall": m[f"recall@{TOP_K}"], "mrr": m["mrr"],
                            "n_retrieved_files": m["n_retrieved_files"],
                            "gen_ms": res.generate_ms, "retrieve_ms": res.retrieve_ms,
                            "n_chunks": len(pipe.chunks), "gen_error": res.error,
                        })
                        tag = f"{codebase}/{strategy}/{method}/{it.id}"
                        print(f"  [gen] {tag}: P={m[f'precision@{TOP_K}']:.2f} "
                              f"R={m[f'recall@{TOP_K}']:.2f} MRR={m['mrr']:.2f} "
                              f"{res.generate_ms:.0f}ms", flush=True)
                    except Exception as exc:  # keep the grid going
                        print(f"  [gen] ERROR {codebase}/{strategy}/{method}/{it.id}: {exc}", flush=True)
                        runs.append({
                            "codebase": codebase, "strategy": strategy, "method": method,
                            "qid": it.id, "tags": it.tags, "gold_files": it.gold_files,
                            "question": it.question, "reference": it.reference_answer,
                            "answer": "", "cited_files": [], "retrieved_files": [],
                            "precision": 0.0, "recall": 0.0, "mrr": 0.0,
                            "n_retrieved_files": 0.0, "gen_ms": 0.0, "retrieve_ms": 0.0,
                            "n_chunks": 0, "gen_error": str(exc),
                        })
    return runs


def phase_b_judge(runs: list[dict]) -> None:
    judge = judge_from_config(CONFIG)
    print(f"[judge] grading {len(runs)} answers with {CONFIG['ollama']['judge_model']} ...", flush=True)
    for i, r in enumerate(runs, 1):
        if not r["answer"]:
            r["judge_score"] = 0
            r["judge_reasoning"] = "no answer"
            r["judge_ms"] = 0.0
            r["judge_error"] = r.get("gen_error")
            continue
        t0 = time.perf_counter()
        j = judge.grade(r["question"], r["reference"], r["answer"])
        r["judge_score"] = j.score
        r["judge_reasoning"] = j.reasoning
        r["judge_ms"] = (time.perf_counter() - t0) * 1000
        r["judge_error"] = j.error
        print(f"  [judge {i}/{len(runs)}] {r['codebase']}/{r['strategy']}/{r['method']}/"
              f"{r['qid']}: score={j.score}", flush=True)


def write_outputs(runs: list[dict]) -> None:
    out_dir = resolve_path(CONFIG["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "grid_runs.jsonl", "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cols = ["codebase", "strategy", "method", "qid", "tags", "n_gold",
            "precision", "recall", "mrr", "n_retrieved_files",
            "judge_score", "gen_ms", "retrieve_ms", "judge_ms", "n_chunks",
            "gen_error", "judge_error"]
    with open(out_dir / "grid_results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({
                "codebase": r["codebase"], "strategy": r["strategy"], "method": r["method"],
                "qid": r["qid"], "tags": ";".join(r["tags"]), "n_gold": len(r["gold_files"]),
                "precision": round(r["precision"], 4), "recall": round(r["recall"], 4),
                "mrr": round(r["mrr"], 4), "n_retrieved_files": int(r["n_retrieved_files"]),
                "judge_score": r.get("judge_score", 0),
                "gen_ms": round(r["gen_ms"], 1), "retrieve_ms": round(r["retrieve_ms"], 1),
                "judge_ms": round(r.get("judge_ms", 0.0), 1), "n_chunks": r["n_chunks"],
                "gen_error": r.get("gen_error") or "", "judge_error": r.get("judge_error") or "",
            })
    print(f"[done] wrote {out_dir/'grid_results.csv'} and grid_runs.jsonl", flush=True)


def main() -> None:
    start = time.perf_counter()
    runs = phase_a_generate()
    phase_b_judge(runs)
    write_outputs(runs)
    mins = (time.perf_counter() - start) / 60
    n_ok = sum(1 for r in runs if not r.get("gen_error"))
    print(f"[done] {len(runs)} runs ({n_ok} ok) in {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
