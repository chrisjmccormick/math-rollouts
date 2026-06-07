"""MATH-500 native id <-> math12k unique_id bridge.

MATH-500 ships native ids (``test/geometry/627.json``); the rollout cohorts key on
math12k ids (``train/geometry/9467``, a row index into the competition_math pool).
The bridge is recoverable two independent, agreeing ways:

  * by problem TEXT (normalize whitespace+case, match) against the math12k table —
    this is ``native_to_math12k`` from the source ``build_qwenmath_generations.py``;
  * by reading the two id columns already present in ``math500_passK.parquet``.

``build_mapping`` derives one and (when both sources are available) asserts they
agree, then ``write_mapping`` emits the canonical ``mappings/math500_to_math12k``
{json,csv} artifact for the dataset.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_norm = lambda t: re.sub(r"\s+", " ", str(t).strip().lower())


def mapping_from_passk(passk_parquet: str | Path) -> dict[str, str]:
    """native_id -> math12k unique_id, read from a parquet that carries both
    ``math500_native_id`` and ``unique_id`` columns (e.g. math500_passK)."""
    import pandas as pd

    df = pd.read_parquet(passk_parquet, columns=["unique_id", "math500_native_id"])
    df = df.dropna(subset=["math500_native_id"]).drop_duplicates()
    return dict(zip(df.math500_native_id.astype(str), df.unique_id.astype(str)))


def mapping_by_text(math_problems_parquet: str | Path) -> dict[str, str]:
    """native_id -> math12k unique_id by normalized problem-text match against the
    canonical math12k table (``math_problems.parquet``). Mirrors the source
    ``native_to_math12k``."""
    import pandas as pd

    from .problems import load_math500

    ds = pd.read_parquet(math_problems_parquet, columns=["unique_id", "problem"])
    text2id = {_norm(p): u for p, u in zip(ds.problem, ds.unique_id)}
    out = {}
    for prob in load_math500():
        key = _norm(prob["problem"])
        if key in text2id:
            out[prob["unique_id"]] = text2id[key]
    return out


def build_mapping(*, passk_parquet: str | Path | None = None,
                  math_problems_parquet: str | Path | None = None) -> dict[str, str]:
    """Build the native->math12k mapping. If BOTH sources are given, assert they
    agree (a faithfulness cross-check) and return the text-derived map."""
    by_text = mapping_by_text(math_problems_parquet) if math_problems_parquet else None
    by_passk = mapping_from_passk(passk_parquet) if passk_parquet else None
    if by_text and by_passk:
        common = set(by_text) & set(by_passk)
        disagree = {k: (by_text[k], by_passk[k]) for k in common if by_text[k] != by_passk[k]}
        if disagree:
            raise ValueError(f"text vs passk mapping disagree on {len(disagree)} ids: "
                             f"{dict(list(disagree.items())[:3])}")
        # union (text is authoritative where both exist; they agree on the overlap)
        return {**by_passk, **by_text}
    result = by_text or by_passk
    if result is None:
        raise ValueError("provide passk_parquet and/or math_problems_parquet")
    return result


def write_mapping(mapping: dict[str, str], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``math500_to_math12k.{json,csv}`` into ``out_dir`` (sorted by native id)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(mapping.items())
    jpath = out_dir / "math500_to_math12k.json"
    cpath = out_dir / "math500_to_math12k.csv"
    jpath.write_text(json.dumps(dict(items), indent=2, ensure_ascii=False))
    with cpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["math500_native_id", "unique_id"])
        w.writerows(items)
    return jpath, cpath


class MathIdMapper:
    """Bidirectional lookup over a native<->math12k mapping (from JSON or dict)."""

    def __init__(self, mapping: dict[str, str]):
        self.native_to_math12k = dict(mapping)
        self.math12k_to_native = {v: k for k, v in mapping.items()}

    @classmethod
    def from_json(cls, path: str | Path) -> "MathIdMapper":
        return cls(json.loads(Path(path).read_text()))

    def to_math12k(self, native_id: str) -> str | None:
        return self.native_to_math12k.get(native_id)

    def to_native(self, math12k_id: str) -> str | None:
        return self.math12k_to_native.get(math12k_id)
