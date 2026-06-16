"""Compute agreement between the local LLM judge and manual grading.

Reads eval/judge_validation.csv (a stratified subset of the grid, manually
re-graded against the same 1-5 rubric) and reports exact agreement, within-one
agreement, mean absolute difference and the direction of disagreements.

    python eval/validate_judge.py
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV = Path(__file__).parent / "judge_validation.csv"


def main() -> None:
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
    n = len(rows)
    exact = within1 = abs_sum = judge_higher = manual_higher = 0
    for r in rows:
        j, m = int(r["judge_score"]), int(r["manual_score"])
        d = j - m
        abs_sum += abs(d)
        exact += d == 0
        within1 += abs(d) <= 1
        judge_higher += d > 0
        manual_higher += d < 0
    print(f"Sample size:            {n}")
    print(f"Exact agreement:        {exact}/{n}  ({100*exact/n:.0f}%)")
    print(f"Within +/-1:            {within1}/{n}  ({100*within1/n:.0f}%)")
    print(f"Mean absolute diff:     {abs_sum/n:.2f}")
    print(f"Judge higher than human:{judge_higher}   Human higher than judge: {manual_higher}")
    if judge_higher and not manual_higher:
        print("Direction: the judge is mildly lenient (never scored below the manual grade).")


if __name__ == "__main__":
    main()
