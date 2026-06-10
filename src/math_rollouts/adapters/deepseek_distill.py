"""Thinking-mode adapter for DeepSeek-R1-Distill-Qwen models.

Same shape as ``qwen3_think``: the prompt ends inside a forced ``<think>\\n`` so the
nucleus is over the first REASONING token, branch expansion stops at ``</think>``,
and correctness is scored on the post-``</think>`` region only. Per DeepSeek's usage
recommendations there is no system prompt; the boxed-answer instruction rides on the
user turn.

NOTE the R1-distill reasoning tokens are ``<think>``=151648 / ``</think>``=151649 —
NOT Qwen3's 151667/151668, despite the shared Qwen2 vocabulary base. The literal
``</think>`` STRING is identical, so string-based scoring (``post-think-v1`` /
``check_correct_post_think``) works unchanged; the ids matter for ``terminal_ids``
(branch expansion) and prompt construction.

The HF chat template changed across revisions: the original Jan-2025 release ends the
generation prompt at ``<｜Assistant｜>``, later revisions append ``<think>\\n``
themselves. ``prompt_ids`` forces ``<think>\\n`` only when the template didn't, so
the rendered prompt (and hence ``prompt_num_tokens``) is identical either way.
"""
from __future__ import annotations

from .base import ModelAdapter

THINK_OPEN = 151648     # "<think>"
THINK_CLOSE = 151649    # "</think>"
NL = 198                # "\n"
USER_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _think_open_ids(tok) -> list[int]:
    ids = tok("<think>\n", add_special_tokens=False).input_ids
    assert ids == [THINK_OPEN, NL], (
        f"unexpected <think>\\n encoding {ids}; expected {[THINK_OPEN, NL]} — wrong tokenizer/model?")
    return ids


class DeepseekR1DistillAdapter(ModelAdapter):
    is_thinking = True

    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.model_id = model_id

    def prompt_ids(self, problem: dict, tok) -> list[int]:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": problem["problem"] + USER_SUFFIX}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
        )
        ids = list(chat)
        opener = _think_open_ids(tok)
        if ids[-len(opener):] != opener:    # older template: stops at <｜Assistant｜>
            ids += opener
        return ids

    def terminal_ids(self, tok) -> dict[int, str]:
        return {tok.eos_token_id: "eos", THINK_CLOSE: "</think>"}

    def score(self, completion_text: str, answer: str) -> bool:
        from ..score.scorers import check_correct_post_think
        return check_correct_post_think(completion_text, answer)

    def vllm_stop(self) -> list[str]:
        return []
