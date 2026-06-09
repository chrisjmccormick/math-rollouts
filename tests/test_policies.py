"""Opener-policy table: coverage pre-pass + the four policy formulas."""
from __future__ import annotations

import pandas as pd

from math_rollouts.analysis.policies import (
    missing_openers,
    per_opener_accuracy,
    policy_table,
)


def _nuclei():
    return pd.DataFrame([
        dict(unique_id="p1", subject="Algebra", branch_path=[0], nuc_prob=0.6),
        dict(unique_id="p1", subject="Algebra", branch_path=[1], nuc_prob=0.4),
        dict(unique_id="p2", subject="Geometry", branch_path=[0], nuc_prob=1.0),
    ])


def _scored_full():
    return pd.DataFrame([
        dict(unique_id="p1", branch_path=[0], answer_matches=True),
        dict(unique_id="p1", branch_path=[0], answer_matches=False),
        dict(unique_id="p1", branch_path=[1], answer_matches=True),
        dict(unique_id="p2", branch_path=[0], answer_matches=True),
    ])


def test_missing_openers_detects_gap():
    nuclei = _nuclei()
    # p1[1] has no rollouts -> reported missing.
    scored = _scored_full()[lambda d: ~((d.unique_id == "p1") & (d.branch_path.map(tuple) == (1,)))]
    acc = per_opener_accuracy(scored)
    assert missing_openers(nuclei, acc) == [("p1", (1,))]


def test_full_coverage_has_no_missing():
    acc = per_opener_accuracy(_scored_full())
    assert missing_openers(_nuclei(), acc) == []


def test_policy_table_warns_on_gap(capsys):
    nuclei = _nuclei()
    scored = _scored_full()[lambda d: ~((d.unique_id == "p1") & (d.branch_path.map(tuple) == (1,)))]
    table = policy_table(nuclei, scored)
    out = capsys.readouterr().out
    assert "WARNING" in out and "p1[1]" in out
    # missing opener folds in as accuracy 0: probability = 0.6*0.5 + 0.4*0 = 0.3.
    p1 = table[table.unique_id == "p1"].iloc[0]
    assert abs(p1.probability - 0.3) < 1e-9
    assert abs(p1.oracle - 0.5) < 1e-9


def test_policy_table_silent_on_full_coverage(capsys):
    policy_table(_nuclei(), _scored_full())
    assert "WARNING" not in capsys.readouterr().out
