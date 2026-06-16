"""Answer generation with a local coder model (Qwen2.5-Coder) via Ollama.

The model is instructed to answer strictly from the retrieved context and to
cite source files in square brackets. Citations are parsed back out of the
answer; if the model omits them, the retrieved files are used as a fallback so
the UI always has something to link to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from .retrieval.base import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a senior software engineer helping a developer understand an "
    "unfamiliar codebase. Answer the question using ONLY the numbered code "
    "context provided. Cite every source file you rely on by its path in square "
    "brackets, e.g. [app/models.py]. If the context does not contain the answer, "
    "say so plainly instead of guessing. Be concise, accurate and technical."
)

_FENCE = {"typescript": "ts", "python": "python", "html": "html",
          "css": "css", "scss": "scss", "angular": "ts", "javascript": "js"}
_CITE = re.compile(r"\[([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)\]")


@dataclass
class AnswerResult:
    answer: str
    cited_files: list[str]
    retrieved_files: list[str]
    model: str
    n_context: int
    prompt_chars: int = 0
    error: str | None = None


def build_context(retrieved: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, rc in enumerate(retrieved, 1):
        meta = rc.metadata or {}
        loc = f"lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        fence = _FENCE.get(meta.get("language", ""), "")
        blocks.append(f"[{i}] File: {rc.rel_path} ({loc})\n```{fence}\n{rc.text}\n```")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, base_url: str, model: str, temperature: float = 0.1,
                 num_ctx: int = 8192, max_tokens: int = 600,
                 repeat_penalty: float = 1.15, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.timeout = timeout

    def answer(self, question: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
        context = build_context(retrieved)
        retrieved_files: list[str] = []
        for rc in retrieved:
            for f in rc.files:
                if f not in retrieved_files:
                    retrieved_files.append(f)

        user = (f"Code context:\n\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the context above, and cite the files you used "
                "in [square brackets].")
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens,
                        "repeat_penalty": self.repeat_penalty,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            return AnswerResult("", [], retrieved_files, self.model, len(retrieved),
                                len(user), error=str(exc))

        cited = []
        for m in _CITE.findall(text):
            norm = m.replace("\\", "/")
            if norm not in cited:
                cited.append(norm)
        return AnswerResult(
            answer=text,
            cited_files=cited or retrieved_files[:3],
            retrieved_files=retrieved_files,
            model=self.model,
            n_context=len(retrieved),
            prompt_chars=len(user),
        )


def generator_from_config(config: dict) -> Generator:
    o, g = config["ollama"], config["generation"]
    return Generator(
        base_url=o["base_url"],
        model=o["gen_model"],
        temperature=g.get("temperature", 0.1),
        num_ctx=g.get("num_ctx", 8192),
        max_tokens=g.get("max_tokens", 600),
        repeat_penalty=g.get("repeat_penalty", 1.15),
        timeout=o.get("request_timeout", 600),
    )
