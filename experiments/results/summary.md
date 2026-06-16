# Experiment results

## Per-configuration summary

| Codebase | Chunking | Retrieval | Precision@k | Recall@k | MRR | Judge | Gen (s) |
|---|---|---|---|---|---|---|---|
| formfit | naive | dense | 0.14 | 0.23 | 0.32 | 3.50 | 6.1 |
| formfit | naive | hybrid | 0.16 | 0.27 | 0.46 | 3.90 | 6.1 |
| formfit | component | dense | 0.05 | 0.10 | 0.26 | 3.70 | 6.4 |
| formfit | component | hybrid | 0.08 | 0.16 | 0.42 | 3.40 | 6.2 |
| formfit | ast | dense | 0.18 | 0.33 | 0.40 | 3.70 | 5.7 |
| formfit | ast | hybrid | 0.25 | 0.41 | 0.47 | 3.60 | 6.4 |
| microblog | naive | dense | 0.60 | 0.68 | 0.83 | 3.60 | 6.1 |
| microblog | naive | hybrid | 0.40 | 0.66 | 0.87 | 3.70 | 5.4 |
| microblog | component | dense | 0.44 | 0.57 | 0.75 | 3.50 | 5.5 |
| microblog | component | hybrid | 0.43 | 0.63 | 0.85 | 3.90 | 5.6 |
| microblog | ast | dense | 0.52 | 0.78 | 0.85 | 4.00 | 5.1 |
| microblog | ast | hybrid | 0.42 | 0.68 | 0.88 | 3.80 | 5.3 |

## Averaged by chunking strategy

| Chunking | Precision@k | Recall@k | MRR | Judge |
|---|---|---|---|---|
| naive | 0.33 | 0.46 | 0.62 | 3.67 |
| component | 0.25 | 0.36 | 0.57 | 3.62 |
| ast | 0.34 | 0.55 | 0.65 | 3.77 |

## Averaged by retrieval method

| Retrieval | Precision@k | Recall@k | MRR | Judge |
|---|---|---|---|---|
| dense | 0.32 | 0.45 | 0.57 | 3.67 |
| hybrid | 0.29 | 0.47 | 0.66 | 3.72 |

**Best configuration by judge score:** microblog / ast / dense (judge 4.00, recall 0.78).
