#!/usr/bin/env python3
"""Rebuild the canonical math12k problem pool ``problems/math_problems.parquet``.

Deterministic: pulls the ~9 MB ``qwedsacf/competition_math`` pool (12,500 rows,
the original Hendrycks MATH train+test bundled with NO split labels) and annotates
each row with a STABLE ``unique_id = train/<subject_slug>/<source_idx>`` where
``source_idx`` is the row index into that HF dataset. This is exactly the key the
migrated rollout parquets join on, so the pool fully describes their problems.

Split labels: ``math500`` = normalized-text match against ``HuggingFaceH4/MATH-500``
(the training holdouts), ``train``/``test`` = the 7500 source-index boundary.

Ported from the source ``build_math_dataset.py`` (the one-off that originally built
the gitignored table). Re-run only if the upstream mirrors change.

    python scripts/build_problems_math12k.py --out-root /path/to/math-rollouts-data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from math_rollouts.data.problems import (
    _normalize_subject,
    _parse_level,
    extract_boxed_answer,
    subject_short,
)

POOL_DATASET_ID = "qwedsacf/competition_math"
MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
TRAIN_TEST_BOUNDARY = 7500

# Full title-case subject -> underscore slug (the historical unique_id form).
SUBJECT_SLUG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}

SCHEMA = pa.schema([
    ("unique_id", pa.string()),
    ("source_idx", pa.int32()),
    ("split", pa.string()),
    ("subject", pa.string()),
    ("subj", pa.string()),
    ("level", pa.int16()),
    ("problem", pa.string()),
    ("solution", pa.string()),
    ("answer", pa.string()),
])


def _subject_slug(subject: str) -> str:
    return SUBJECT_SLUG.get(subject, subject.lower().replace(" ", "_"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    from datasets import load_dataset

    src = load_dataset(POOL_DATASET_ID, split="train")
    math500_norms = {_norm(r["problem"]) for r in load_dataset(MATH500_DATASET_ID, split="test")}

    rows, matched = [], 0
    for i, row in enumerate(src):
        subject = _normalize_subject(row.get("type", row.get("subject", "")))
        solution = row.get("solution", "")
        answer = row.get("answer") or extract_boxed_answer(solution)
        if not answer:
            continue                              # ~4 rows with no boxed answer
        if _norm(row["problem"]) in math500_norms:
            split = "math500"; matched += 1
        elif i >= TRAIN_TEST_BOUNDARY:
            split = "test"
        else:
            split = "train"
        rows.append({
            "unique_id": f"train/{_subject_slug(subject)}/{i}",
            "source_idx": i, "split": split, "subject": subject,
            "subj": subject_short(subject), "level": _parse_level(row.get("level", -1)),
            "problem": row["problem"], "solution": solution, "answer": answer,
        })

    # invariants (mirror the source builder).
    n_m500 = sum(r["split"] == "math500" for r in rows)
    assert n_m500 == 500, f"expected 500 math500 rows, got {n_m500}"
    assert matched == 500, f"matched {matched} / 500 MATH-500 holdouts"
    assert len({r["unique_id"] for r in rows}) == len(rows), "unique_id collision"

    table = pa.table({n: [r.get(n) for r in rows] for n in SCHEMA.names}, schema=SCHEMA)
    out_dir = Path(a.out_root) / "problems"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "math_problems.parquet"
    pq.write_table(table, out_path)
    print(f"wrote {out_path}: {len(rows)} problems, {n_m500} math500 holdouts")


if __name__ == "__main__":
    main()
