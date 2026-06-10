"""Thinking-mode adapter for Qwen3-8B (post-trained reasoning model).

The chat template (``enable_thinking=True``) ends at the assistant turn without
emitting ``<think>`` itself, so we force ``<think>\\n`` and take the nucleus of the
first REASONING token as the opener. Branch expansion stops at ``</think>``;
correctness is scored on the post-``</think>`` region only.

Ported from the source ``openers/lib/think_prompt.py``. NOTE the Qwen3 reasoning
close token is 151668 — NOT the R1-distill 151649 that some tooling defaults to.
"""
from __future__ import annotations

from .base import ModelAdapter

THINK_OPEN = 151667     # "<think>"
THINK_CLOSE = 151668    # "</think>"
NL = 198                # "\n"
USER_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _think_open_ids(tok) -> list[int]:
    ids = tok("<think>\n", add_special_tokens=False).input_ids
    assert ids == [THINK_OPEN, NL], (
        f"unexpected <think>\\n encoding {ids}; expected {[THINK_OPEN, NL]} — wrong tokenizer/model?")
    return ids


class Qwen3ThinkAdapter(ModelAdapter):
    is_thinking = True

    def __init__(self, model_id: str = "Qwen/Qwen3-8B"):
        self.model_id = model_id

    def prompt_ids(self, problem: dict, tok) -> list[int]:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": problem["problem"] + USER_SUFFIX}],
            add_generation_prompt=True, enable_thinking=True,
            tokenize=True, return_dict=False,
        )
        return list(chat) + _think_open_ids(tok)

    def terminal_ids(self, tok) -> dict[int, str]:
        return {tok.eos_token_id: "eos", THINK_CLOSE: "</think>"}

    def score(self, completion_text: str, answer: str) -> bool:
        from ..score.scorers import check_correct_post_think
        return check_correct_post_think(completion_text, answer)

    def vllm_stop(self) -> list[str]:
        return []

    def sampling_overrides(self) -> dict:
        # Vendor thinking-mode sampling (generation_config.json): T=0.6, top_p=0.95,
        # top_k=20. T/top_p already match the project defaults; top_k is extra — and
        # the legacy qwen3-8b pools were sampled WITH it (rows carry top_k=20), so
        # extension batches must keep it or the pool mixes distributions.
        return {"top_k": 20}
