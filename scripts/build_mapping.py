#!/usr/bin/env python3
"""Emit the canonical MATH-500 <-> math12k bridge into ``mappings/``.

Builds ``native_id -> math12k unique_id`` from a parquet carrying both columns
(``math500_passK``), and cross-checks against a SECOND independent source — either
the math12k problem table by normalized text (``--problems``), or another passK
parquet (``--cross-passk``) — asserting full agreement before writing
``math500_to_math12k.{json,csv}``.

    python scripts/build_mapping.py \\
        --passk  <snap>/qwen2.5-math-1.5b/math500_passK.parquet \\
        --cross-passk <snap>/qwen2.5-math-1.5b-oat-zero/math500_passK.parquet \\
        --out-root /path/to/math-rollouts-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

from math_rollouts.data.ids import (
    MathIdMapper,
    build_mapping,
    mapping_from_passk,
    write_mapping,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passk", required=True, help="math500_passK.parquet (both id cols)")
    ap.add_argument("--problems", default=None, help="math_problems.parquet (text cross-check)")
    ap.add_argument("--cross-passk", default=None, help="a second passK parquet to cross-check")
    ap.add_argument("--out-root", required=True, help="dataset root; writes mappings/ under it")
    a = ap.parse_args()

    mapping = build_mapping(passk_parquet=a.passk, math_problems_parquet=a.problems)

    if a.cross_passk:
        other = mapping_from_passk(a.cross_passk)
        common = set(mapping) & set(other)
        disagree = {k: (mapping[k], other[k]) for k in common if mapping[k] != other[k]}
        if disagree:
            raise SystemExit(f"cross-passk disagreement on {len(disagree)}: "
                             f"{dict(list(disagree.items())[:3])}")
        print(f"[build_mapping] cross-passk agrees on all {len(common)} shared ids")

    # invariants: 500 entries, bijection, known spot-check.
    assert len(mapping) == 500, f"expected 500 entries, got {len(mapping)}"
    assert len(set(mapping.values())) == len(mapping), "mapping is not a bijection"
    m = MathIdMapper(mapping)
    assert m.to_math12k("test/geometry/627.json") == "train/geometry/9467", \
        f"geometry/627 spot-check failed: {m.to_math12k('test/geometry/627.json')}"

    jpath, cpath = write_mapping(mapping, Path(a.out_root) / "mappings")
    print(f"[build_mapping] wrote {jpath} and {cpath} ({len(mapping)} entries, bijection OK)")


if __name__ == "__main__":
    main()
