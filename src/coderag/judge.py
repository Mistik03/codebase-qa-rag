"""LLM-as-judge answer grading with a local Llama-3.1-8B-Instruct model.

Manually grading every answer in the experiment grid is infeasible, so a local
judge scores each answer 1-5 for correctness and grounding against a reference
answer. The judge is itself validated against manual grades on a subset (see
``eval/``) so the thesis can report how trustworthy it is.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

JUDGE_SYSTEM = (
    "You are a strict but fair grader of answers about source code. You compare a "
    "candidate answer to a reference answer for the same question and judge how "
    "correct and well-grounded the candidate is."
)

JUDGE_RUBRIC = (
    "Score on a 1-5 integer scale:\n"
    "5 = fully correct and complete, consistent with the reference.\n"
    "4 = correct with minor omissions.\n"
    "3 = partially correct; misses or muddles important points.\n"
    "2 = mostly incorrect or largely ungrounded.\n"
    "1 = wrong, irrelevant, or empty.\n"
    'Respond with ONLY a JSON object: {"score": <int 1-5>, "reasoning": "<one sentence>"}.'
)


@dataclass
class Judgement:
    score: int
    reasoning: str
    raw: str = ""
    error: str | None = None


class Judge:
    def __init__(self, base_url: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def grade(self, question: str, reference: str, candidate: str) -> Judgement:
        user = (f"Question:\n{question}\n\n"
                f"Reference answer:\n{reference}\n\n"
                f"Candidate answer:\n{candidate or '(empty)'}\n\n{JUDGE_RUBRIC}")
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0, "num_ctx": 8192},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            return Judgement(0, "", error=str(exc))
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Judgement:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                num = re.search(r"[1-5]", raw)
                return Judgement(int(num.group()) if num else 0, raw, raw)
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return Judgement(0, "", raw, error="unparseable judge output")
        try:
            score = int(round(float(data.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(5, score))
        return Judgement(score, str(data.get("reasoning", "")).strip(), raw)


def judge_from_config(config: dict) -> Judge:
    o = config["ollama"]
    return Judge(o["base_url"], o["judge_model"], o.get("request_timeout", 600))
