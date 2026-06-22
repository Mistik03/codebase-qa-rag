"""Paired significance tests on the per-question results.

For each comparison we pair observations by question (codebase + qid), averaging
over the nuisance dimension (e.g. when comparing chunking strategies we average
each question's metric over the two retrieval methods), then run a paired
(dependent) t-test with scipy. Recall@k, the LLM-judge score and MRR are tested.

A paired t-test is what the supervisor requested; with a 40-question benchmark
its normality assumption is only approximate, so results are read as indicative
and the small sample is noted as a limitation in the thesis.

    python -m experiments.stats
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy import stats

from coderag.config import CONFIG, resolve_path

RESULTS = resolve_path(CONFIG["paths"]["results_dir"])
RUNS = RESULTS / "grid_runs.jsonl"

METRICS = [("recall", "Recall@k"), ("judge_score", "Judge"), ("mrr", "MRR")]
DIM = {"chunking": "strategy", "retrieval": "method"}
COMPARISONS = [
    ("chunking", "ast", "naive"),
    ("chunking", "ast", "component"),
    ("chunking", "naive", "component"),
    ("retrieval", "hybrid", "dense"),
]


def load_runs() -> list[dict]:
    return [json.loads(line) for line in open(RUNS, encoding="utf-8") if line.strip()]


def collect(runs: list[dict], dim: str, metric: str) -> dict[str, dict[tuple, float]]:
    """dim value -> {(codebase, qid): metric averaged over the other dimension}."""
    acc: dict[tuple, list[float]] = {}
    for r in runs:
        acc.setdefault((r[dim], (r["codebase"], r["qid"])), []).append(float(r[metric]))
    out: dict[str, dict[tuple, float]] = {}
    for (value, q), vals in acc.items():
        out.setdefault(value, {})[q] = sum(vals) / len(vals)
    return out


def paired_ttest(a_map: dict, b_map: dict):
    keys = sorted(set(a_map) & set(b_map))
    a = [a_map[k] for k in keys]
    b = [b_map[k] for k in keys]
    try:
        res = stats.ttest_rel(a, b)
        t, p = float(res.statistic), float(res.pvalue)
    except Exception:
        t, p = float("nan"), float("nan")
    mean_a = sum(a) / len(a) if a else float("nan")
    mean_b = sum(b) / len(b) if b else float("nan")
    return len(keys), mean_a, mean_b, t, p


def main() -> None:
    runs = load_runs()
    rows = []
    for kind, a, b in COMPARISONS:
        dim = DIM[kind]
        for metric, label in METRICS:
            coll = collect(runs, dim, metric)
            if a not in coll or b not in coll:
                continue
            n, ma, mb, t, p = paired_ttest(coll[a], coll[b])
            rows.append({
                "comparison": f"{a} vs {b}", "dimension": kind, "metric": label,
                "n": n, f"mean_{a}": round(ma, 3), f"mean_{b}": round(mb, 3),
                "mean_diff": round(ma - mb, 3), "t": round(t, 3),
                "p_value": round(p, 4), "significant_0.05": "yes" if p < 0.05 else "no",
            })

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "dimension", "metric", "n", "mean_a", "mean_b",
                    "mean_diff", "t", "p_value", "significant_0.05"])
        for r in rows:
            a, b = r["comparison"].split(" vs ")
            w.writerow([r["comparison"], r["dimension"], r["metric"], r["n"],
                        r[f"mean_{a}"], r[f"mean_{b}"], r["mean_diff"], r["t"],
                        r["p_value"], r["significant_0.05"]])

    lines = ["# Paired t-test results", "",
             "Paired by question (codebase + qid), averaged over the other dimension. "
             "Significance at alpha = 0.05.", "",
             "| Comparison | Metric | N | mean A | mean B | diff (A-B) | t | p | sig? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["comparison"].split(" vs ")
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['n']} | {r[f'mean_{a}']} | "
                     f"{r[f'mean_{b}']} | {r['mean_diff']} | {r['t']} | {r['p_value']} | "
                     f"{r['significant_0.05']} |")
    n_sig = sum(1 for r in rows if r["significant_0.05"] == "yes")
    lines += ["", f"Statistically significant comparisons (p < 0.05): {n_sig} of {len(rows)}."]
    (RESULTS / "stats.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[done] wrote stats.csv and stats.md ({n_sig}/{len(rows)} significant)")


if __name__ == "__main__":
    main()
