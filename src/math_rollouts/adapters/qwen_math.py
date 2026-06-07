"""Non-thinking adapter for the Qwen2.5-Math family (base + Oat-Zero RL ckpt).

Uses the literal Qwen-Math template (boxed instruction in the SYSTEM turn, bare
problem in the USER turn) built as a raw string — NOT ``apply_chat_template`` — for
byte-exact parity with the source ``run_random_nothink`` / ``openings_k16`` recipe.
The whole completion is the answer region, so correctness is the full-text boxed
match.
"""
from __future__ import annotations

from .base import ModelAdapter

QWEN_MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def apply_qwen_math_template(user_body: str) -> str:
    return (
        "<|im_start|>system\n" + QWEN_MATH_SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + user_body + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class QwenMathAdapter(ModelAdapter):
    is_thinking = False

    def __init__(self, model_id: str = "Qwen/Qwen2.5-Math-1.5B"):
        self.model_id = model_id

    def prompt_ids(self, problem: dict, tok) -> list[int]:
        text = apply_qwen_math_template(problem["problem"])
        return tok(text, add_special_tokens=False).input_ids

    def terminal_ids(self, tok) -> dict[int, str]:
        return {tok.eos_token_id: "eos"}

    def score(self, completion_text: str, answer: str) -> bool:
        from ..score.scorers import check_correct
        return check_correct(completion_text, answer)

    def vllm_stop(self) -> list[str]:
        return ["<|im_end|>"]
