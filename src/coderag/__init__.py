"""coderag — a fully local Retrieval-Augmented Generation system for codebase Q&A.

All models run on the developer's own machine through Ollama; no source code is
ever sent to an external service. The package is organised around three
swappable axes that form the experimental grid of the thesis:

  * chunking strategy : naive | component-aware | AST-aware
  * retrieval method  : dense | hybrid (dense + BM25 via reciprocal-rank fusion)
  * generation model  : a small quantized coder model (Qwen2.5-Coder)
"""

__version__ = "0.1.0"
