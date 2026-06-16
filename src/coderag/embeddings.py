"""Text embeddings via a local Ollama model (``nomic-embed-text``).

nomic-embed-text is trained with task prefixes: documents are embedded with a
``search_document:`` prefix and queries with ``search_query:``. Using the right
prefix on each side measurably improves retrieval, so it is applied here rather
than left to the caller. Requests go to Ollama's ``/api/embed`` endpoint (batch)
with a per-item fallback for older daemons.
"""
from __future__ import annotations

import requests


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, document_prefix: str = "",
                 query_prefix: str = "", batch_size: int = 16, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + t for t in texts[i:i + self.batch_size]]
            out.extend(self._embed(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    # -- transport ---------------------------------------------------------------
    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"]
        except requests.RequestException:
            pass
        # Fallback: older /api/embeddings accepts a single prompt at a time.
        out: list[list[float]] = []
        for text in inputs:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        return out


def embedder_from_config(config: dict) -> OllamaEmbedder:
    o, e = config["ollama"], config["embedding"]
    return OllamaEmbedder(
        base_url=o["base_url"],
        model=o["embed_model"],
        document_prefix=e.get("document_prefix", ""),
        query_prefix=e.get("query_prefix", ""),
        batch_size=e.get("batch_size", 16),
        timeout=o.get("request_timeout", 600),
    )
