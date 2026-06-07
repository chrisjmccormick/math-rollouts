"""Adapter wiring: registry resolution, prompt construction, thinking vs
non-thinking scoring. No model download — prompt construction is checked at the
template/registry level; scoring uses the real math_verify backend."""
from __future__ import annotations

import pytest

from math_rollouts.adapters import (
    PaperBaseAdapter,
    Qwen3ThinkAdapter,
    QwenMathAdapter,
    get_adapter,
)
from math_rollouts.adapters.paper_base import paper_student_text
from math_rollouts.adapters.qwen_math import apply_qwen_math_template
from math_rollouts.score.scorers import check_correct, check_correct_post_think, get_scorer


def test_registry_resolution():
    assert isinstance(get_adapter("Qwen/Qwen2.5-Math-1.5B"), QwenMathAdapter)
    assert isinstance(get_adapter("sail/Qwen2.5-Math-1.5B-Oat-Zero"), QwenMathAdapter)
    assert isinstance(get_adapter("Qwen/Qwen3-8B"), Qwen3ThinkAdapter)
    assert isinstance(get_adapter("Qwen/Qwen3-8B-Base"), PaperBaseAdapter)


def test_is_thinking_flags():
    assert get_adapter("Qwen/Qwen2.5-Math-1.5B").is_thinking is False
    assert get_adapter("Qwen/Qwen3-8B").is_thinking is True
    assert get_adapter("Qwen/Qwen3-8B-Base").is_thinking is False


def test_registry_family_override_and_unknown():
    assert isinstance(get_adapter("some/unknown", family="qwen3_think"), Qwen3ThinkAdapter)
    with pytest.raises(KeyError):
        get_adapter("totally/unregistered")


def test_qwen_math_template_shape():
    t = apply_qwen_math_template("What is 2+2?")
    assert t.startswith("<|im_start|>system\n")
    assert "What is 2+2?" in t
    assert t.endswith("<|im_start|>assistant\n")


def test_paper_base_prompt_shape():
    t = paper_student_text("What is 2+2?")
    assert t.startswith("Question: What is 2+2?")
    assert "Let's think step by step." in t


def test_non_thinking_scoring_full_text():
    assert check_correct("The answer is \\boxed{42}.", "42") is True
    assert check_correct("The answer is \\boxed{41}.", "42") is False


def test_thinking_scoring_post_think_only():
    # answer only inside the reasoning -> not counted; after </think> -> counted.
    leaked = "reasoning \\boxed{42} </think> final \\boxed{7}"
    assert check_correct_post_think(leaked, "42") is False
    good = "reasoning ... </think> the answer is \\boxed{42}"
    assert check_correct_post_think(good, "42") is True
    assert check_correct_post_think("no close token \\boxed{42}", "42") is False


def test_default_scorer_gates_on_finish_reason():
    scorer = get_scorer("boxed-match-stop-v1")
    base_row = dict(model_id="M", unique_id="u", run_id=0, branch_path=[0],
                    sample_idx=0, completion_text="x \\boxed{42}", answer="42")
    assert scorer.score_row({**base_row, "finish_reason": "stop"})["is_correct"] is True
    # correct text but truncated (length) -> not a keeper under the default scorer.
    assert scorer.score_row({**base_row, "finish_reason": "length"})["is_correct"] is False


def test_ungated_scorer_ignores_finish_reason():
    scorer = get_scorer("boxed-match-v1")
    row = dict(model_id="M", unique_id="u", run_id=0, branch_path=[0], sample_idx=0,
               completion_text="x \\boxed{42}", answer="42", finish_reason="length")
    assert scorer.score_row(row)["is_correct"] is True
