"""Completion-style adapter for pure base LMs (e.g. Qwen3-8B-Base).

Uses the paper-style completion prompt (no chat template, no ``<think>``):

    Question: {problem}
    Answer: Let's think step by step.

Base models tend to roll into a fresh question after answering, so STOP strings
cut the run-on. The whole completion is the answer region (full-text boxed match).
Ported from the source ``openers/lib/paper_prompt.py``.
"""
from __future__ import annotations

from .base import ModelAdapter

PRIMER = "Answer: Let's think step by step."
STOP = ["\nQuestion:", "\nProblem:", "[Question", " Question:"]


def paper_student_text(problem: str) -> str:
    return f"Question: {problem}\n{PRIMER}"


class PaperBaseAdapter(ModelAdapter):
    is_thinking = False

    def __init__(self, model_id: str = "Qwen/Qwen3-8B-Base"):
        self.model_id = model_id

    def prompt_ids(self, problem: dict, tok) -> list[int]:
        return tok(paper_student_text(problem["problem"]), add_special_tokens=False).input_ids

    def terminal_ids(self, tok) -> dict[int, str]:
        return {tok.eos_token_id: "eos"}

    def score(self, completion_text: str, answer: str) -> bool:
        from ..score.scorers import check_correct
        return check_correct(completion_text, answer)

    def vllm_stop(self) -> list[str]:
        return list(STOP)
