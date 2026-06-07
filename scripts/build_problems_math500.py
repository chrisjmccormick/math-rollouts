#!/usr/bin/env python3
"""Build the self-contained ``problems/math500.parquet`` table.

One row per MATH-500 problem: native id, the math12k ``unique_id`` (via the
mappings artifact), subject/level, problem/solution text, and the boxed answer.
Self-contained — problem text comes from the official ``HuggingFaceH4/MATH-500``
mirror, so the dataset needs no external pool to describe its MATH-500 inputs.

    python scripts/build_problems_math500.py \\
        --mapping /path/to/math-rollouts-data/mappings/math500_to_math12k.json \\
        --out-root /path/to/math-rollouts-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from math_rollouts.data.ids import MathIdMapper
from math_rollouts.data.problems import load_math500

SCHEMA = pa.schema([
    ("math500_native_id", pa.string()),
    ("unique_id", pa.string()),          # math12k id (nullable if unmapped)
    ("subject", pa.string()),
    ("subj", pa.string()),
    ("level", pa.int16()),
    ("problem", pa.string()),
    ("solution", pa.string()),
    ("answer", pa.string()),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", required=True, help="math500_to_math12k.json")
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    mapper = MathIdMapper.from_json(a.mapping)
    problems = load_math500()
    rows = []
    for p in problems:
        rows.append({
            "math500_native_id": p["unique_id"],
            "unique_id": mapper.to_math12k(p["unique_id"]),
            "subject": p["subject"],
            "subj": p["subj"],
            "level": int(p["level"]),
            "problem": p["problem"],
            "solution": p["solution"],
            "answer": p["answer"],
        })
    n_mapped = sum(r["unique_id"] is not None for r in rows)
    table = pa.table({n: [r.get(n) for r in rows] for n in SCHEMA.names}, schema=SCHEMA)
    out_dir = Path(a.out_root) / "problems"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "math500.parquet"
    pq.write_table(table, out_path)
    print(f"wrote {out_path}: {len(rows)} rows, {n_mapped} mapped to math12k")


if __name__ == "__main__":
    main()
