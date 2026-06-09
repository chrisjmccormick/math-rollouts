"""Flat, self-contained rollout POOLS — assemble, score, count, extend.

A pool is *scored natural rollouts in the canonical schema* (``schema.POOL_SCHEMA`` =
``ROLLOUTS_SCHEMA`` + ``is_correct`` + ``scorer_id``), stored as a single
``generations/<model-slug>/<pool>.parquet`` with an ``is_correct`` inline (labeled by
``scorer_id``) and per-batch provenance in a ``<pool>.meta.json`` sidecar.

CPU only. Generation (``generate.natural.generate_natural``) produces the raw rows;
these helpers score them, conform them to ``POOL_SCHEMA``, and support the
width-extend (top every problem up to a target K) by computing per-problem deficits.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import POOL_SCHEMA, ROLLOUT_KEY, table_from_rows


def default_scorer_id(model_id: str) -> str:
    """Canonical scorer for a model family: post-think for thinking models (so
    'did </think> close' is handled by the scorer, not a schema column), else the
    boxed-match gate."""
    from ..adapters import get_adapter
    return "post-think-v1" if get_adapter(model_id).is_thinking else "boxed-match-stop-v1"


def score_rollouts(rows: list[dict], *, scorer_id: str | None = None,
                   model_id: str | None = None) -> tuple[list[bool], str]:
    """Score raw rollout rows in memory. Returns ``(is_correct_list, scorer_id)``.
    ``scorer_id`` defaults to the model's canonical scorer."""
    from ..score.scorers import get_scorer
    if not rows:
        return [], scorer_id or "boxed-match-stop-v1"
    scorer_id = scorer_id or default_scorer_id(model_id or rows[0]["model_id"])
    scorer = get_scorer(scorer_id)
    return [bool(scorer.score_row(r)["is_correct"]) for r in rows], scorer_id


def to_pool_frame(rows: list[dict], is_correct: list[bool], scorer_id: str):
    """Conform raw rollout rows + their verdicts to ``POOL_SCHEMA`` (a pandas
    DataFrame). Missing canonical fields are filled null; extra legacy fields dropped."""
    merged = [{**r, "is_correct": bool(c), "scorer_id": scorer_id}
              for r, c in zip(rows, is_correct)]
    return table_from_rows(merged, POOL_SCHEMA).to_pandas()


def build_pool(rows: list[dict], *, scorer_id: str | None = None,
               model_id: str | None = None):
    """Score + conform in one step. Returns ``(pool_df, scorer_id)``."""
    verdicts, sid = score_rollouts(rows, scorer_id=scorer_id, model_id=model_id)
    return to_pool_frame(rows, verdicts, sid), sid


def pool_counts(df, id_col: str = "unique_id"):
    """Rollouts per problem (a pandas Series indexed by ``id_col``)."""
    return df.groupby(id_col).size()


def pool_deficit(df, target_k: int, id_col: str = "unique_id",
                 ids: list[str] | None = None) -> dict[str, int]:
    """Per-problem shortfall to reach ``target_k`` rollouts: ``{uid: target_k - have}``
    for every problem with fewer than ``target_k`` (problems already at/above are
    omitted). If ``ids`` is given, problems absent from ``df`` count as a full
    ``target_k`` deficit (so brand-new problems are generated from scratch)."""
    have = pool_counts(df, id_col).to_dict()
    universe = ids if ids is not None else list(have)
    out = {uid: target_k - int(have.get(uid, 0)) for uid in universe}
    return {uid: n for uid, n in out.items() if n > 0}


def next_run_id(df) -> int:
    """One past the max existing ``run_id`` (0 if the pool is empty/missing it)."""
    if df is None or "run_id" not in getattr(df, "columns", []) or not len(df):
        return 0
    return int(df["run_id"].max()) + 1


def _key_tuples(df):
    bp = df["branch_path"].map(lambda x: tuple(x) if x is not None else ())
    cols = [df[c] if c != "branch_path" else bp for c in ROLLOUT_KEY]
    return list(zip(*cols))


def extend_pool(existing, new):
    """Append ``new`` pool rows to ``existing`` (both ``POOL_SCHEMA`` DataFrames),
    dropping any rows whose ``ROLLOUT_KEY`` already exists (so re-runs are
    idempotent). Returns the combined DataFrame."""
    import pandas as pd
    if existing is None or not len(existing):
        return new.reset_index(drop=True)
    seen = set(_key_tuples(existing))
    mask = [t not in seen for t in _key_tuples(new)]
    return pd.concat([existing, new[mask]], ignore_index=True)[list(existing.columns)]


def is_canonical(df) -> bool:
    """True if ``df`` already carries the canonical pool columns (vs a legacy pool)."""
    return {"scorer_id", "is_correct"} <= set(getattr(df, "columns", []))


def ensure_pool_schema(df, model_id: str | None = None, *, scorer_id: str | None = None):
    """Return ``df`` as a canonical-schema pool: a no-op (column reorder) if already
    canonical, else migrate a legacy pool in place (re-scoring ``is_correct``). Lets
    the extend path work regardless of whether a pool has been migrated yet."""
    if is_canonical(df):
        from ..schema import POOL_SCHEMA
        return df[[c for c in POOL_SCHEMA.names if c in df.columns]]
    pool_df, _ = migrate_legacy_pool(df, scorer_id=scorer_id, model_id=model_id)
    return pool_df


def migrate_legacy_pool(legacy_df, *, scorer_id: str | None = None,
                        model_id: str | None = None):
    """Project a legacy dev-project pool DataFrame onto ``POOL_SCHEMA`` and re-score.

    Drops the baggage columns (kept only ``POOL_SCHEMA`` fields), fills the natural-
    sampling opener fields (``depth=0, branch_path=[], opener_token_ids=[]``), and
    recomputes ``is_correct`` from ``completion_text`` with the canonical scorer.
    Returns ``(pool_df, scorer_id)``."""
    rows = legacy_df.to_dict("records")
    for r in rows:
        r.setdefault("depth", 0)
        r["branch_path"] = list(r.get("branch_path") or [])
        r["opener_token_ids"] = list(r.get("opener_token_ids") or [])
    verdicts, sid = score_rollouts(rows, scorer_id=scorer_id, model_id=model_id)
    return to_pool_frame(rows, verdicts, sid), sid


def pool_drift_report(legacy_df, migrated_df, id_col: str = "unique_id") -> dict:
    """Compare legacy vs re-scored ``is_correct``: per-rollout flip counts and how many
    problems change difficulty band (band = base-model solve-rate bucket)."""
    from ..analysis.difficulty import assign_band
    old = legacy_df["is_correct"].reset_index(drop=True).astype(bool)
    new = migrated_df["is_correct"].reset_index(drop=True).astype(bool)
    flips = (old != new)
    old_band = legacy_df.groupby(id_col)["is_correct"].mean().map(assign_band)
    new_band = migrated_df.groupby(id_col)["is_correct"].mean().map(assign_band)
    band_moved = (old_band != new_band.reindex(old_band.index))
    return {
        "n_rollouts": int(len(old)),
        "n_flips": int(flips.sum()),
        "flip_to_correct": int((~old & new).sum()),
        "flip_to_incorrect": int((old & ~new).sum()),
        "n_problems": int(old_band.size),
        "problems_band_moved": int(band_moved.sum()),
    }


def refresh_shard_is_correct(shard_df, pool_df):
    """Update a ``*_token_nuclei`` shard's copied ``is_correct`` from a (re-scored)
    pool, joining on ``(unique_id, run_id, sample_idx)``. Returns a new DataFrame."""
    key = ["unique_id", "run_id", "sample_idx"]
    truth = pool_df[key + ["is_correct"]].drop_duplicates(key)
    out = shard_df.drop(columns=["is_correct"]).merge(truth, on=key, how="left")
    return out[list(shard_df.columns)]


def write_pool(df, path: str | Path) -> Path:
    """Write a pool DataFrame to parquet, coerced to ``POOL_SCHEMA`` (zstd)."""
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table_from_rows(df.to_dict("records"), POOL_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def write_pool_meta(path: str | Path, *, model_id: str, pool: str, scorer_id: str,
                    gen_config: dict, runs: list[dict], df=None) -> Path:
    """Write the ``<pool>.meta.json`` provenance sidecar next to a pool parquet.
    ``runs`` is a list of per-batch dicts (run_id, cohort, k, seed, n_rollouts, ...)."""
    path = Path(path)
    meta = {
        "model_id": model_id, "pool": pool, "schema": "POOL_SCHEMA",
        "scorer_id": scorer_id, "gen_config": gen_config, "runs": runs,
    }
    if df is not None:
        meta["n_rollouts"] = int(len(df))
        meta["n_problems"] = int(df["unique_id"].nunique())
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
