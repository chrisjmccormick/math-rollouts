"""CPU scoring pass: read raw ``rollouts.parquet`` -> write ``scores.parquet``.

Re-runnable and GPU-free. Different scorers (``--scorer``) produce different score
tables; the raw rollouts are never touched. By default writes
``scores.parquet``; pass ``--out`` to keep multiple scorers side by side.

  python -m math_rollouts.score.run --rollouts <dir>/rollouts.parquet \
      --scorer boxed-match-stop-v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from ..schema import SCORES_SCHEMA, table_from_rows
from .scorers import get_scorer


def score_file(rollouts_path: str | Path, scorer_id: str = "boxed-match-stop-v1",
               out_path: str | Path | None = None, tokenizer_id: str | None = None) -> Path:
    import pandas as pd

    rollouts_path = Path(rollouts_path)
    df = pd.read_parquet(rollouts_path)
    kw = {}
    if scorer_id == "leak-filter-v1" and tokenizer_id:
        from transformers import AutoTokenizer
        kw["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_id)
    scorer = get_scorer(scorer_id, **kw)

    rows = [scorer.score_row(r) for r in df.to_dict("records")]
    table = table_from_rows(rows, SCORES_SCHEMA)
    out_path = Path(out_path) if out_path else rollouts_path.parent / "scores.parquet"
    pq.write_table(table, out_path)
    n_correct = sum(r["is_correct"] for r in rows)
    print(f"wrote {out_path}  ({len(rows)} scores, {n_correct} correct, scorer={scorer_id})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, required=True, help="path to rollouts.parquet")
    ap.add_argument("--scorer", default="boxed-match-stop-v1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id (only needed for leak-filter token fraction)")
    args = ap.parse_args()
    score_file(args.rollouts, args.scorer, args.out, args.tokenizer)


if __name__ == "__main__":
    main()
