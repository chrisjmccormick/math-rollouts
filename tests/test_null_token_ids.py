"""Null ``completion_token_ids`` handling (some legacy guided-rollouts pools never
stored them): token-derived lengths must be NULL, not fabricated 0, and
``dup_index`` must fall back to completion-text identity rather than collapsing
every null-id rollout in a problem onto one shared () identity."""
from __future__ import annotations

import pandas as pd

from math_rollouts.data.pools import add_dup_index, row_attributes


def _row(text="x </think> \\boxed{42}", ids=None, **kw):
    return {"answer": "42", "completion_text": text, "completion_token_ids": ids,
            "finish_reason": "stop", **kw}


def test_null_token_ids_give_null_lengths():
    a = row_attributes(_row(ids=None), eos_id=5, prompt_len=100)
    assert a["completion_num_tokens"] is None
    assert a["total_num_tokens"] is None
    assert a["answer_token_frac"] is None
    assert a["prompt_num_tokens"] == 100          # prompt length is independent
    assert a["terminal"] == "emitted_eos"          # text facts unaffected


def test_nan_token_ids_treated_as_null():
    # pandas surfaces a null list cell as float NaN in row dicts.
    a = row_attributes(_row(ids=float("nan")), eos_id=5)
    assert a["completion_num_tokens"] is None and a["total_num_tokens"] is None


def test_present_token_ids_still_counted():
    a = row_attributes(_row(ids=[1, 2, 3, 5]), eos_id=5, prompt_len=10)
    assert a["completion_num_tokens"] == 3         # trailing EOS excluded
    assert a["total_num_tokens"] == 13


def test_stored_count_wins_over_null_ids():
    a = row_attributes(_row(ids=None, completion_num_tokens=42), eos_id=5)
    assert a["completion_num_tokens"] == 42


def test_dup_index_null_ids_use_text_identity():
    df = pd.DataFrame([
        {"unique_id": "u", "run_id": 0, "sample_idx": 0,
         "completion_token_ids": None, "completion_text": "same"},
        {"unique_id": "u", "run_id": 0, "sample_idx": 1,
         "completion_token_ids": None, "completion_text": "same"},
        {"unique_id": "u", "run_id": 0, "sample_idx": 2,
         "completion_token_ids": None, "completion_text": "different"},
    ])
    d = add_dup_index(df)
    # exact-text repeat flagged; a distinct completion is NOT a duplicate.
    assert list(d["dup_index"]) == [0, 1, 0]


def test_dup_index_mixed_id_and_null_rows():
    df = pd.DataFrame([
        {"unique_id": "u", "run_id": 0, "sample_idx": 0,
         "completion_token_ids": [1, 2], "completion_text": "a"},
        {"unique_id": "u", "run_id": 0, "sample_idx": 1,
         "completion_token_ids": [1, 2], "completion_text": "a"},
        {"unique_id": "u", "run_id": 0, "sample_idx": 2,
         "completion_token_ids": None, "completion_text": "a"},
    ])
    d = add_dup_index(df)
    # token-id identity for id rows; the null-id row is its own (text) identity.
    assert list(d["dup_index"]) == [0, 1, 0]
