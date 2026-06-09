"""Head-to-head banded comparison: per-problem tallies, pooling, markdown render."""
from __future__ import annotations

import pandas as pd
import pytest

from math_rollouts.analysis.bandtable import (
    band_compare,
    per_problem,
    render_markdown,
)


def _rollouts(rows):
    """rows: list of (unique_id, subject, answer_matches)."""
    return pd.DataFrame(rows, columns=["unique_id", "subject", "answer_matches"])


# Two problems: p1 Algebra (Easy), p2 Geometry (Hard).
_BANDS = {"p1": "Easy", "p2": "Hard"}


def _cohort_A():
    # p1: 2/2 correct, p2: 1/2 correct
    return _rollouts([
        ("p1", "Algebra", True), ("p1", "Algebra", True),
        ("p2", "Geometry", True), ("p2", "Geometry", False),
    ])


def _cohort_B():
    # p1: 1/2 correct, p2: 0/2 correct
    return _rollouts([
        ("p1", "Algebra", True), ("p1", "Algebra", False),
        ("p2", "Geometry", False), ("p2", "Geometry", False),
    ])


def test_per_problem_integer_tallies():
    pp = per_problem(_cohort_A())
    p1 = pp[pp.unique_id == "p1"].iloc[0]
    assert (int(p1.n_correct), int(p1.n)) == (2, 2)
    assert pp.n_correct.dtype.kind == "i" and pp.n.dtype.kind == "i"


def test_band_compare_shape_and_band_mapping():
    tidy = band_compare([("A", per_problem(_cohort_A())),
                         ("B", per_problem(_cohort_B()))], _BANDS)
    # 2 problems x 2 cohorts = 4 rows
    assert len(tidy) == 4
    assert set(tidy.band) == {"Easy", "Hard"}
    assert set(tidy.label) == {"A", "B"}


def test_unmapped_problem_folds_into_unknown_band():
    tidy = band_compare([("A", per_problem(_cohort_A()))], {"p1": "Easy"})
    assert (tidy[tidy.unique_id == "p2"].band == "Unknown").all()


def test_render_markdown_pools_bolds_best_and_adds_delta():
    tidy = band_compare([("A", per_problem(_cohort_A())),
                         ("B", per_problem(_cohort_B()))], _BANDS)
    md = render_markdown(tidy, ["A", "B"], title="demo")
    # auto Δ column for exactly two cohorts
    assert "Δ (A − B)" in md
    # Easy/Algebra: A is 2/2=100% (bolded best), B is 1/2=50%
    assert "**(2 / 2) 100.0%**" in md
    # Overall: A=3/4=75%, B=1/4=25%, Δ=+50.0pp
    assert "+50.0pp" in md
    # band headers present in canonical order (Easy before Hard)
    assert md.index("**Easy**") < md.index("**Hard**")


def _mixed_batches():
    """One problem with a 2-rollout baseline (run_id=0) + a 4-rollout pass@k
    expansion (run_id=1) at a much lower success rate."""
    rows = [("p1", "Algebra", True, 0), ("p1", "Algebra", True, 0)]
    rows += [("p1", "Algebra", False, 1)] * 4
    return pd.DataFrame(rows, columns=["unique_id", "subject", "answer_matches", "run_id"])


def test_per_problem_refuses_to_pool_mixed_run_ids():
    with pytest.raises(ValueError, match=r"run_id"):
        per_problem(_mixed_batches())


def test_per_problem_run_ids_selects_single_batch():
    # baseline only: 2/2 correct (the pass@k expansion is excluded)
    pp = per_problem(_mixed_batches(), run_ids=0)
    row = pp[pp.unique_id == "p1"].iloc[0]
    assert (int(row.n_correct), int(row.n)) == (2, 2)


def test_per_problem_run_ids_list_pools_deliberately():
    # explicit pooling of both batches: 2/6
    pp = per_problem(_mixed_batches(), run_ids=[0, 1])
    row = pp[pp.unique_id == "p1"].iloc[0]
    assert (int(row.n_correct), int(row.n)) == (2, 6)


def test_per_problem_single_run_id_needs_no_selection():
    df = _mixed_batches()
    df = df[df.run_id == 0]  # only one run_id present -> no guard trip
    pp = per_problem(df)
    assert int(pp.iloc[0].n) == 2


def test_render_markdown_no_delta_for_three_cohorts():
    tidy = band_compare([("A", per_problem(_cohort_A())),
                         ("B", per_problem(_cohort_B())),
                         ("C", per_problem(_cohort_A()))], _BANDS)
    md = render_markdown(tidy, ["A", "B", "C"])
    assert "Δ" not in md
