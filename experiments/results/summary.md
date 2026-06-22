# Experiment results

## Per-configuration summary

| Codebase | Chunking | Retrieval | Precision@k | Recall@k | MRR | Judge | Gen (s) |
|---|---|---|---|---|---|---|---|
| formfit | naive | dense | 0.15 | 0.29 | 0.33 | 3.60 | 6.0 |
| formfit | naive | hybrid | 0.18 | 0.34 | 0.54 | 3.70 | 5.4 |
| formfit | component | dense | 0.08 | 0.23 | 0.38 | 3.50 | 6.2 |
| formfit | component | hybrid | 0.13 | 0.34 | 0.62 | 3.45 | 6.5 |
| formfit | ast | dense | 0.22 | 0.43 | 0.55 | 3.65 | 6.5 |
| formfit | ast | hybrid | 0.28 | 0.53 | 0.71 | 3.75 | 5.7 |
| microblog | naive | dense | 0.52 | 0.74 | 0.81 | 3.65 | 5.3 |
| microblog | naive | hybrid | 0.41 | 0.75 | 0.88 | 3.85 | 5.4 |
| microblog | component | dense | 0.46 | 0.68 | 0.77 | 3.70 | 5.5 |
| microblog | component | hybrid | 0.42 | 0.73 | 0.87 | 3.90 | 5.0 |
| microblog | ast | dense | 0.51 | 0.85 | 0.84 | 3.85 | 5.6 |
| microblog | ast | hybrid | 0.44 | 0.78 | 0.87 | 3.85 | 5.4 |

## Averaged by chunking strategy

| Chunking | Precision@k | Recall@k | MRR | Judge |
|---|---|---|---|---|
| naive | 0.31 | 0.53 | 0.64 | 3.70 |
| component | 0.27 | 0.49 | 0.66 | 3.64 |
| ast | 0.37 | 0.65 | 0.74 | 3.77 |

## Averaged by retrieval method

| Retrieval | Precision@k | Recall@k | MRR | Judge |
|---|---|---|---|---|
| dense | 0.32 | 0.54 | 0.61 | 3.66 |
| hybrid | 0.31 | 0.58 | 0.75 | 3.75 |

**Best configuration by judge score:** microblog / component / hybrid (judge 3.90, recall 0.73).
