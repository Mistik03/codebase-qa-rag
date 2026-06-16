"""Aggregate the experiment grid into tables and figures.

Reads ``experiments/results/grid_results.csv`` (produced by ``run_grid``) and
writes:
  summary_by_config.csv     mean metrics per (codebase, strategy, method)
  summary.md                markdown tables for the thesis
  fig_recall_by_strategy.png, fig_judge_by_config.png,
  fig_method_comparison.png, fig_latency.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from coderag.config import CONFIG, resolve_path

RESULTS = resolve_path(CONFIG["paths"]["results_dir"])
STRATEGY_ORDER = ["naive", "component", "ast"]
METHOD_ORDER = ["dense", "hybrid"]
COLORS = {"dense": "#4C72B0", "hybrid": "#DD8452",
          "microblog": "#55A868", "formfit": "#C44E52"}


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "grid_results.csv")
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGY_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["codebase", "strategy", "method"], observed=True)
             .agg(precision=("precision", "mean"),
                  recall=("recall", "mean"),
                  mrr=("mrr", "mean"),
                  judge=("judge_score", "mean"),
                  gen_ms=("gen_ms", "mean"),
                  n=("qid", "count"))
             .reset_index())
    return agg.sort_values(["codebase", "strategy", "method"])


def _grouped_bar(ax, df, index_col, group_col, value_col, order_index, order_group, title, ylabel):
    idx = [i for i in order_index if i in set(df[index_col])]
    groups = [g for g in order_group if g in set(df[group_col])]
    width = 0.8 / max(1, len(groups))
    for gi, g in enumerate(groups):
        sub = df[df[group_col] == g].set_index(index_col)[value_col]
        vals = [sub.get(i, 0) for i in idx]
        xs = [x + gi * width for x in range(len(idx))]
        ax.bar(xs, vals, width=width, label=str(g), color=COLORS.get(g))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([x + width * (len(groups) - 1) / 2 for x in range(len(idx))])
    ax.set_xticklabels(idx)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def figures(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    # 1) recall@k by chunking strategy, grouped by codebase (avg over methods)
    by_strat_cb = (df.groupby(["strategy", "codebase"], observed=True)["recall"]
                     .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bar(ax, by_strat_cb, "strategy", "codebase", "recall",
                 STRATEGY_ORDER, ["microblog", "formfit"],
                 "Mean recall@k by chunking strategy", "recall@k")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_recall_by_strategy.png", dpi=150); plt.close(fig)

    # 2) judge score by strategy, grouped by retrieval method (avg over codebases)
    by_strat_method = (df.groupby(["strategy", "method"], observed=True)["judge_score"]
                         .mean().reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _grouped_bar(ax, by_strat_method, "strategy", "method", "judge_score",
                 STRATEGY_ORDER, METHOD_ORDER,
                 "Mean LLM-judge score by strategy and retrieval method", "judge score (1-5)")
    ax.set_ylim(0, 5.2)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_judge_by_config.png", dpi=150); plt.close(fig)

    # 3) dense vs hybrid on recall and MRR (avg over everything)
    method_cmp = df.groupby("method", observed=True)[["recall", "mrr"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    metrics = ["recall", "mrr"]
    width = 0.35
    for mi, method in enumerate(METHOD_ORDER):
        if method not in method_cmp.index:
            continue
        vals = [method_cmp.loc[method, m] for m in metrics]
        xs = [x + mi * width for x in range(len(metrics))]
        ax.bar(xs, vals, width=width, label=method, color=COLORS.get(method))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([x + width / 2 for x in range(len(metrics))])
    ax.set_xticklabels(["recall@k", "MRR"])
    ax.set_title("Dense vs hybrid retrieval"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_method_comparison.png", dpi=150); plt.close(fig)

    # 4) mean generation latency by strategy
    lat = df.groupby("strategy", observed=True)["gen_ms"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len([s for s in STRATEGY_ORDER if s in lat.index]))
    vals = [lat.get(s, 0) / 1000 for s in STRATEGY_ORDER if s in lat.index]
    ax.bar(list(xs), vals, color="#8172B3")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.05, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([s for s in STRATEGY_ORDER if s in lat.index])
    ax.set_title("Mean answer-generation latency (Qwen2.5-Coder 3B)"); ax.set_ylabel("seconds")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_latency.png", dpi=150); plt.close(fig)


def markdown_tables(agg: pd.DataFrame, df: pd.DataFrame) -> str:
    lines = ["# Experiment results\n"]
    lines.append("## Per-configuration summary\n")
    lines.append("| Codebase | Chunking | Retrieval | Precision@k | Recall@k | MRR | Judge | Gen (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        lines.append(f"| {r.codebase} | {r.strategy} | {r.method} | {r.precision:.2f} | "
                     f"{r.recall:.2f} | {r.mrr:.2f} | {r.judge:.2f} | {r.gen_ms/1000:.1f} |")

    lines.append("\n## Averaged by chunking strategy\n")
    by_s = df.groupby("strategy", observed=True).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        mrr=("mrr", "mean"), judge=("judge_score", "mean")).reindex(STRATEGY_ORDER).dropna()
    lines.append("| Chunking | Precision@k | Recall@k | MRR | Judge |")
    lines.append("|---|---|---|---|---|")
    for s, r in by_s.iterrows():
        lines.append(f"| {s} | {r.precision:.2f} | {r.recall:.2f} | {r.mrr:.2f} | {r.judge:.2f} |")

    lines.append("\n## Averaged by retrieval method\n")
    by_m = df.groupby("method", observed=True).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        mrr=("mrr", "mean"), judge=("judge_score", "mean")).reindex(METHOD_ORDER).dropna()
    lines.append("| Retrieval | Precision@k | Recall@k | MRR | Judge |")
    lines.append("|---|---|---|---|---|")
    for m, r in by_m.iterrows():
        lines.append(f"| {m} | {r.precision:.2f} | {r.recall:.2f} | {r.mrr:.2f} | {r.judge:.2f} |")

    best = agg.loc[agg["judge"].idxmax()]
    lines.append(f"\n**Best configuration by judge score:** {best.codebase} / "
                 f"{best.strategy} / {best.method} (judge {best.judge:.2f}, recall {best.recall:.2f}).")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load()
    agg = summarise(df)
    agg.to_csv(RESULTS / "summary_by_config.csv", index=False)
    md = markdown_tables(agg, df)
    (RESULTS / "summary.md").write_text(md, encoding="utf-8")
    figures(df, agg)
    print(md)
    print(f"[done] wrote summary_by_config.csv, summary.md and 4 figures to {RESULTS}")


if __name__ == "__main__":
    main()
