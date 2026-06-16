# Code-QA Benchmark

A small, hand-labelled benchmark for evaluating retrieval-augmented codebase
question-answering. It is released as one of the outcomes of the thesis so that
future work — particularly on the **Angular** ecosystem, which is
under-represented in code-RAG research — can build on it.

## Format

One JSON object per line (`<codebase>.jsonl`):

| Field | Meaning |
|-------|---------|
| `id` | Stable identifier (`mb-01`, `ff-01`, …) |
| `question` | A realistic developer question about the codebase |
| `reference_answer` | A concise correct answer, used by the LLM judge |
| `gold_files` | Files (paths relative to the codebase root) that *must* be retrieved to answer; used for precision/recall/MRR |
| `tags` | Topic labels (`auth`, `routing`, `api`, …) |

## Datasets

| File | Codebase | Language | Questions |
|------|----------|----------|-----------|
| `microblog.jsonl` | [Microblog](https://github.com/miguelgrinberg/microblog) | Python / Flask | 10 |
| `formfit.jsonl` | FormFit (internship project) | TypeScript / Angular | 10 |

`gold_files` are validated against the actual indexed files with
`coderag.benchmark.validate_benchmark`, so every label is reachable by retrieval.

## Labelling method

Questions were authored by reading the source directly and choosing the files a
developer would genuinely need to open to answer each one. Reference answers
describe the real mechanism (e.g. *Supabase-backed `AuthService` with a
`currentUser$` BehaviorSubject*, or *the `followers` self-referential
many-to-many table*) rather than generic explanations.
