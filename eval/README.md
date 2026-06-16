# Judge validation

The experiment grid grades answer quality with a local LLM judge
(Llama 3.1 8B Instruct). Because an LLM judge is itself fallible, a stratified
subset of answers was re-graded **manually** against the same 1–5 rubric and
compared with the judge.

- `judge_validation.csv` — 12 answers spanning both codebases, all three
  chunking strategies, both retrieval methods and a range of judge scores, with
  the judge's score, the manual score and a one-line note on each.
- `validate_judge.py` — recomputes the agreement statistics from that CSV.

Run it with `python eval/validate_judge.py`.

## Summary of the subset

The judge agreed with the manual grade exactly in 8 of 12 cases (67%) and within
one point in all 12 (100%). Every disagreement was the judge scoring *higher*
than the manual grade — a mild leniency toward fluent but poorly-grounded
answers, concentrated in the FormFit cases where retrieval recall was zero. The
practical consequence, discussed in the thesis, is that absolute answer-quality
scores should be read as indicative and the sharpest comparisons rest on the
retrieval metrics.
