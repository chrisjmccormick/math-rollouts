"""MATH-500 evaluation loader + subject shorthand + boxed-answer extraction.

Ported from the source project's ``load_math.py`` but trimmed to the
eval-relevant surface and made self-contained: MATH-500 is read from the official
HuggingFace mirror (``HuggingFaceH4/MATH-500``), which carries native ids
(``test/<subject>/<n>.json``), so no local annotated table is required.
"""
from __future__ import annotations

import re

MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"

_SUBJECT_NORMALIZE = {
    "algebra": "Algebra",
    "counting & probability": "Counting & Probability",
    "counting and probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate algebra": "Intermediate Algebra",
    "number theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}

# Canonical subject shorthand — single source of truth.
SUBJECT_SHORT = {
    "Algebra": "alg",
    "Counting & Probability": "cp",
    "Geometry": "geo",
    "Intermediate Algebra": "ia",
    "Number Theory": "nt",
    "Prealgebra": "pa",
    "Precalculus": "pc",
}
SLUG_SHORT = {
    "algebra": "alg",
    "counting_and_probability": "cp",
    "geometry": "geo",
    "intermediate_algebra": "ia",
    "number_theory": "nt",
    "prealgebra": "pa",
    "precalculus": "pc",
}
SHORT_SUBJECTS = tuple(SUBJECT_SHORT.values())


def _normalize_subject(raw: str) -> str:
    return _SUBJECT_NORMALIZE.get(raw.strip().lower(), raw.strip())


def subject_short(value: str) -> str:
    """Map a full name, slug, short code, OR a unique_id to its short code."""
    if "/" in value:                          # unique_id -> middle segment
        parts = value.split("/")
        value = parts[1] if len(parts) >= 2 else value
    v = value.strip()
    if v.isdigit():
        return "gsm8k"
    if v in SUBJECT_SHORT:
        return SUBJECT_SHORT[v]
    low = v.lower()
    if low in SLUG_SHORT:
        return SLUG_SHORT[low]
    if low in SHORT_SUBJECTS:
        return low
    return SUBJECT_SHORT.get(_normalize_subject(v), v)


def _parse_level(raw) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        m = re.search(r"\d+", raw)
        if m:
            return int(m.group(0))
    return -1


def extract_boxed_answer(solution: str) -> str:
    """Contents of the final ``\\boxed{...}`` in a solution (balanced-brace scan)."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(solution) and depth > 0:
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i]
        i += 1
    return ""


def _row_to_problem(row: dict) -> dict | None:
    subject = _normalize_subject(row.get("subject", ""))
    solution = row.get("solution", "")
    answer = row.get("answer") or extract_boxed_answer(solution)
    if not answer:
        return None
    return {
        "unique_id": row["unique_id"],           # native test/<subject>/<n>.json
        "subject": subject,
        "subj": subject_short(subject),
        "level": _parse_level(row.get("level", -1)),
        "problem": row["problem"],
        "solution": solution,
        "answer": answer,
    }


def load_math500(*, dataset_id: str = MATH500_DATASET_ID, split: str = "test") -> list[dict]:
    """Load the full MATH-500 benchmark (native ids) for generation/eval."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    out = [p for p in (_row_to_problem(r) for r in ds) if p is not None]
    print(f"[math_rollouts] loaded {len(out)} MATH-500 problems "
          f"from {dataset_id}:{split} ({len(ds)} rows)", flush=True)
    return out


def _table_problem(r) -> dict:
    """One problems-table row (itertuples) -> the problem dict generation expects."""
    return {"unique_id": r.unique_id, "subject": r.subject, "subj": r.subj,
            "level": int(r.level), "problem": r.problem,
            "solution": getattr(r, "solution", ""), "answer": r.answer}


def load_problems_by_split(split: str, **kw) -> list[dict]:
    """Problems for one split (``train`` | ``test`` | ``math500``) from the dataset's
    own ``problems/math_problems.parquet`` (split-aware ``unique_id`` + ``split``).
    This — not the external HF MATH-500 mirror — is the canonical generation source."""
    from .hf import load_problems_parquet
    df = load_problems_parquet("math_problems", **kw)
    return [_table_problem(r) for r in df[df["split"] == split].itertuples()]


def load_problems_by_ids(ids: list[str], **kw) -> list[dict]:
    """Problems for explicit split-aware ``unique_id``s from ``math_problems.parquet``."""
    from .hf import load_problems_parquet
    want = set(ids)
    df = load_problems_parquet("math_problems", **kw)
    out = [_table_problem(r) for r in df[df.unique_id.isin(want)].itertuples()]
    missing = want - {p["unique_id"] for p in out}
    if missing:
        print(f"[math_rollouts] warning: {len(missing)} ids not in math_problems "
              f"(first: {sorted(missing)[:3]})", flush=True)
    return out


def load_math500_by_ids(ids: list[str], *, dataset_id: str = MATH500_DATASET_ID,
                        split: str = "test") -> list[dict]:
    """Fetch a MATH-500 subset by native unique_id."""
    if not ids:
        return []
    from datasets import load_dataset

    wanted = set(ids)
    ds = load_dataset(dataset_id, split=split)
    out = [p for p in (_row_to_problem(r) for r in ds
                       if r.get("unique_id") in wanted) if p is not None]
    missing = wanted - {p["unique_id"] for p in out}
    if missing:
        print(f"[math_rollouts] warning: {len(missing)} requested ids not found "
              f"(first: {sorted(missing)[:3]})", flush=True)
    return out
