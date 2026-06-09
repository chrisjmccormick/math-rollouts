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


def test_answer_match_is_truncation_tolerant():
    # DEFAULT scorer: correct iff answer_matches, regardless of how it terminated.
    scorer = get_scorer("answer-match")
    base_row = dict(model_id="M", unique_id="u", run_id=0, branch_path=[0],
                    sample_idx=0, completion_text="x \\boxed{42}", answer="42")
    assert scorer.score_row({**base_row, "finish_reason": "stop"})["verdict"] == "correct"
    # truncated but the answer matched -> still correct (no termination gate).
    assert scorer.score_row({**base_row, "finish_reason": "length"})["verdict"] == "correct"


def test_boxed_match_requires_a_box():
    scorer = get_scorer("boxed-match")
    boxed = dict(model_id="M", unique_id="u", run_id=0, branch_path=[0], sample_idx=0,
                 completion_text="x \\boxed{42}", answer="42", finish_reason="length")
    assert scorer.score_row(boxed)["verdict"] == "correct"
    # right answer but NO box -> incorrect under the box-gated scorer.
    nobox = {**boxed, "completion_text": "the answer is 42"}
    r = scorer.score_row(nobox)
    assert r["answer_matches"] is True and r["has_boxed"] is False
    assert r["verdict"] == "incorrect"


def test_benchmark_budget_gates_on_termination_and_strict_raises():
    import pytest as _pytest
    base = dict(model_id="M", unique_id="u", run_id=0, branch_path=[0], sample_idx=0,
                completion_text="x \\boxed{42}", answer="42")
    sc = get_scorer("benchmark@budget=8192")
    # correct AND naturally terminated -> correct.
    assert sc.score_row({**base, "finish_reason": "stop", "max_gen_len": 3000})["verdict"] == "correct"
    # truncated below budget -> unresolved; strict mode raises.
    with _pytest.raises(ValueError, match="UNRESOLVED|unresolved|budget"):
        sc.score_row({**base, "finish_reason": "length", "max_gen_len": 3000})
    # non-strict records 'unresolved' instead of raising.
    sc_lax = get_scorer("benchmark@budget=8192", strict=False)
    assert sc_lax.score_row({**base, "finish_reason": "length",
                             "max_gen_len": 3000})["verdict"] == "unresolved"
    # truncated but the pool already met the budget -> a clean incorrect.
    assert sc.score_row({**base, "finish_reason": "length",
                         "max_gen_len": 8192})["verdict"] == "incorrect"
