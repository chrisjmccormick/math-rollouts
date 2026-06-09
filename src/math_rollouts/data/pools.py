"""Flat, self-contained rollout POOLS — assemble, annotate, count, extend.

A pool is *natural rollouts in the canonical schema* (``schema.POOL_SCHEMA`` =
``ROLLOUTS_SCHEMA`` + the criterion-free answer/match facts + ``dup_index``), stored
as a single ``generations/<model-slug>/<pool>.parquet`` with per-batch provenance in
a ``<pool>.meta.json`` sidecar. There is NO baked ``is_correct`` boolean: the pool
records the *facts* (``answer_matches``, ``has_boxed``, answer placement, termination,
lengths). Accuracy / difficulty bands are reproduced by a NAMED scorer over those
facts (default ``answer-match`` == the ``answer_matches`` column).

CPU only. Generation (``generate.natural.generate_natural``) produces the raw rows;
these helpers compute the answer/match attributes, conform to ``POOL_SCHEMA``, assign
``dup_index``, and support the width-extend (top every problem up to a target K).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import POOL_SCHEMA, ROLLOUT_KEY, table_from_rows

# Columns added on top of the raw rollout facts to make a pool.
_POOL_ATTRS = ["answer_matches", "has_boxed", "answer_char_pos", "answer_token_frac",
               "terminal", "stop_reason", "completion_num_tokens", "prompt_num_tokens",
               "total_num_tokens"]


def default_scorer_id(model_id: str) -> str:
    """The default REPORTING scorer for a model family (provenance only — the pool
    bakes no verdict). ``post-think-v1`` for thinking models (so 'did </think> close'
    is handled by the scorer), else the permissive ``answer-match``."""
    from ..adapters import get_adapter
    return "post-think-v1" if get_adapter(model_id).is_thinking else "answer-match"


def _eos_excluded_count(token_ids, eos_id) -> int:
    """Response length excluding a single trailing EOS token (vLLM returns EOS as the
    final id of ``completion_token_ids``; a truncated rollout has none)."""
    n = len(token_ids)
    if eos_id is not None and n and token_ids[-1] == eos_id:
        return n - 1
    return n


def _is_thinking(model_id) -> bool:
    """Whether ``model_id``'s adapter is a thinking model (False if unset)."""
    if not model_id:
        return False
    from ..adapters import get_adapter
    return bool(get_adapter(model_id).is_thinking)


def row_attributes(row: dict, *, tok=None, prompt_len=None, eos_id=None,
                   is_thinking: bool = False) -> dict:
    """Criterion-free raw attributes for ONE rollout row. Pure (no shared state), so
    it is reused both serially and by the parallel migration workers.

    Prefers facts already present on the row (e.g. ``terminal``/lengths stamped at
    generation) and computes the rest: ``answer_matches``, ``has_boxed``, the
    verified-answer char position + token fraction, the termination enum, and the
    EOS-excluded lengths. ``prompt_len`` is this problem's ``prompt_num_tokens``.

    For ``is_thinking`` models the *committed* answer is the post-``</think>`` region,
    so ``answer_matches``/``has_boxed``/placement are computed THERE (making
    ``answer_matches`` == the model's default ``post-think-v1`` verdict, which the
    analysis stack reads directly). A thinking completion that never closed ``</think>``
    (e.g. truncated mid-thought) has no committed answer → ``answer_matches`` False.
    Non-thinking models score the whole completion (region == full text)."""
    from ..analysis.positional import has_boxed, verified_answer_char_pos
    from ..score.scorers import (THINK_CLOSE_STR, answer_matches as _answer_matches,
                                 derive_terminal)

    text, answer = row["completion_text"], row["answer"]
    # Scoring region: post-</think> for thinking models (empty if it never closed),
    # else the whole completion. ``pos`` is reported as a FULL-text char offset.
    region_start = 0
    if is_thinking:
        i = text.find(THINK_CLOSE_STR)
        region_start = i + len(THINK_CLOSE_STR) if i != -1 else len(text)
    region = text[region_start:]

    am = bool(_answer_matches(region, answer))
    hb = bool(has_boxed(region))
    rpos = verified_answer_char_pos(region, answer)
    pos = region_start + rpos if rpos is not None else None
    cti = row.get("completion_token_ids")
    ids = list(cti) if cti is not None else []

    comp_n = row.get("completion_num_tokens")
    if comp_n is None:
        comp_n = _eos_excluded_count(ids, eos_id)
    comp_n = int(comp_n)

    p_n = row.get("prompt_num_tokens")
    if p_n is None:
        p_n = prompt_len
    p_n = int(p_n) if p_n is not None else None

    frac = row.get("answer_token_frac")
    if frac is None and pos is not None and tok is not None:
        n_before = len(tok.encode(text[:pos], add_special_tokens=False))
        frac = n_before / max(comp_n, 1)

    terminal = row.get("terminal") or derive_terminal(row.get("finish_reason"),
                                                       row.get("stop_reason"))
    total_n = row.get("total_num_tokens")
    if total_n is None and p_n is not None:
        total_n = p_n + comp_n

    return {
        "answer_matches": am,
        "has_boxed": hb,
        "answer_char_pos": int(pos) if pos is not None else None,
        "answer_token_frac": float(frac) if frac is not None else None,
        "terminal": terminal,
        "stop_reason": row.get("stop_reason"),
        "completion_num_tokens": comp_n,
        "prompt_num_tokens": p_n,
        "total_num_tokens": int(total_n) if total_n is not None else None,
    }


def compute_raw_attributes(rows: list[dict], *, tok=None,
                           prompt_len: dict | None = None, eos_id=None,
                           is_thinking: bool = False) -> list[dict]:
    """Annotate raw rollout rows with the criterion-free pool attributes (serial).
    ``prompt_len`` maps ``unique_id -> prompt_num_tokens``. ``is_thinking`` routes the
    answer/match facts to the post-``</think>`` region (see ``row_attributes``)."""
    pl = prompt_len or {}
    return [{**r, **row_attributes(r, tok=tok, prompt_len=pl.get(r.get("unique_id")),
                                   eos_id=eos_id, is_thinking=is_thinking)} for r in rows]


def add_dup_index(df):
    """Assign ``dup_index``: 0 for the first occurrence of an identical completion
    within a problem, 1,2,... for natural repeats. Identity is the exact
    ``completion_token_ids`` sequence (the precise notion of a distinct completion).
    Deterministic order by ``(unique_id, run_id, sample_idx)``. Duplicate completions
    are KEPT (legitimate near-deterministic behaviour + needed for pass@k); the flag
    lets analysis filter to distinct completions (``dup_index == 0``)."""
    d = df.copy()
    d["_ord"] = range(len(d))
    d["_cid"] = d["completion_token_ids"].map(
        lambda x: tuple(int(t) for t in x) if x is not None else ())
    sort_cols = [c for c in ("unique_id", "run_id", "sample_idx") if c in d.columns]
    d = d.sort_values(sort_cols + ["_ord"], kind="stable")
    d["dup_index"] = d.groupby(["unique_id", "_cid"]).cumcount().astype("int32")
    d = d.sort_values("_ord", kind="stable").drop(columns=["_ord", "_cid"])
    return d.reset_index(drop=True)


def to_pool_frame(rows: list[dict]):
    """Conform annotated rollout rows to ``POOL_SCHEMA`` (a pandas DataFrame), filling
    ``dup_index`` if not already present on the rows. Rows must already carry the raw
    attributes (run ``compute_raw_attributes`` / ``row_attributes`` first)."""
    df = table_from_rows(rows, POOL_SCHEMA).to_pandas()
    if "dup_index" not in df.columns or df["dup_index"].isna().all():
        df = add_dup_index(df)
    return df


def build_pool(rows: list[dict], *, model_id: str | None = None, tok=None,
               prompt_len: dict | None = None, eos_id=None):
    """Annotate raw rollout rows + conform to ``POOL_SCHEMA`` in one step."""
    annotated = compute_raw_attributes(rows, tok=tok, prompt_len=prompt_len, eos_id=eos_id,
                                       is_thinking=_is_thinking(model_id))
    return to_pool_frame(annotated)


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
    idempotent). Returns the combined DataFrame. NOTE: this de-dups on the rollout
    KEY (same problem/run/sample), NOT on completion content — duplicate completions
    across distinct samples are deliberately kept (see ``add_dup_index``)."""
    import pandas as pd
    if existing is None or not len(existing):
        return new.reset_index(drop=True)
    seen = set(_key_tuples(existing))
    mask = [t not in seen for t in _key_tuples(new)]
    return pd.concat([existing, new[mask]], ignore_index=True)[list(existing.columns)]


def is_canonical(df) -> bool:
    """True if ``df`` already carries the canonical pool facts (vs a legacy pool)."""
    return "answer_matches" in set(getattr(df, "columns", []))


def ensure_pool_schema(df, model_id: str | None = None, *, tok=None,
                       prompt_len: dict | None = None, eos_id=None):
    """Return ``df`` as a canonical-schema pool: a no-op (column reorder) if already
    canonical, else migrate a legacy pool (compute the answer/match attributes). Lets
    the extend path work regardless of whether a pool has been migrated yet."""
    if is_canonical(df):
        return df[[c for c in POOL_SCHEMA.names if c in df.columns]]
    return migrate_legacy_pool(df, model_id=model_id, tok=tok, prompt_len=prompt_len,
                               eos_id=eos_id)


def migrate_legacy_pool(legacy_df, *, model_id: str | None = None, tok=None,
                        prompt_len: dict | None = None, eos_id=None):
    """Project a legacy pool DataFrame onto ``POOL_SCHEMA`` and compute the raw
    answer/match attributes + ``dup_index``.

    Drops the baggage columns (keeps only ``POOL_SCHEMA`` fields), fills the natural-
    sampling opener fields (``depth=0, branch_path=[], opener_token_ids=[]``), and
    derives ``answer_matches``/``has_boxed``/placement/termination/lengths from
    ``completion_text`` + the vLLM fields. A pre-existing ``dup_index`` (e.g. on an
    already-deduped-flag pool) is preserved. Returns a pandas DataFrame."""
    rows = legacy_df.to_dict("records")
    for r in rows:
        r.setdefault("depth", 0)
        bp, op = r.get("branch_path"), r.get("opener_token_ids")
        # explicit None-checks: empty numpy arrays are ambiguous under `or`.
        r["branch_path"] = list(bp) if bp is not None else []
        r["opener_token_ids"] = list(op) if op is not None else []
    annotated = compute_raw_attributes(rows, tok=tok, prompt_len=prompt_len, eos_id=eos_id,
                                       is_thinking=_is_thinking(model_id))
    df = table_from_rows(annotated, POOL_SCHEMA).to_pandas()
    if "dup_index" in legacy_df.columns:
        df["dup_index"] = legacy_df["dup_index"].astype("int32").to_numpy()
    else:
        df = add_dup_index(df)
    return df


def pool_drift_report(legacy_df, migrated_df, id_col: str = "unique_id") -> dict:
    """Compare legacy ``is_correct`` vs the re-derived ``answer_matches``: per-rollout
    flip counts and how many problems change difficulty band (band = base-model
    solve-rate bucket under the default ``answer-match`` verdict)."""
    from ..analysis.difficulty import assign_band
    old = legacy_df["is_correct"].reset_index(drop=True).astype(bool)
    new = migrated_df["answer_matches"].reset_index(drop=True).astype(bool)
    flips = (old != new)
    old_band = legacy_df.groupby(id_col)["is_correct"].mean().map(assign_band)
    new_band = migrated_df.groupby(id_col)["answer_matches"].mean().map(assign_band)
    band_moved = (old_band != new_band.reindex(old_band.index))
    return {
        "n_rollouts": int(len(old)),
        "n_flips": int(flips.sum()),
        "flip_to_correct": int((~old & new).sum()),
        "flip_to_incorrect": int((old & ~new).sum()),
        "n_problems": int(old_band.size),
        "problems_band_moved": int(band_moved.sum()),
    }


def refresh_shard_answer_matches(shard_df, pool_df):
    """Update a ``*_token_nuclei`` shard's copied correctness from a (re-derived) pool:
    replace the legacy ``is_correct`` column with ``answer_matches`` and attach
    ``dup_index``, joining on ``(unique_id, run_id, sample_idx)``. Returns a new
    DataFrame preserving the shard's column order (``is_correct`` slot becomes
    ``answer_matches``; ``dup_index`` appended)."""
    key = ["unique_id", "run_id", "sample_idx"]
    truth = pool_df[key + ["answer_matches", "dup_index"]].drop_duplicates(key)
    base = shard_df.drop(columns=[c for c in ("is_correct", "answer_matches", "dup_index")
                                  if c in shard_df.columns])
    out = base.merge(truth, on=key, how="left")
    order = []
    for c in shard_df.columns:
        if c == "is_correct":
            order.append("answer_matches")
        elif c in ("answer_matches", "dup_index"):
            continue
        else:
            order.append(c)
    if "answer_matches" not in order:
        order.append("answer_matches")
    order.append("dup_index")
    return out[order]


def write_pool(df, path: str | Path) -> Path:
    """Write a pool DataFrame to parquet, coerced to ``POOL_SCHEMA`` (zstd)."""
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table_from_rows(df.to_dict("records"), POOL_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def write_pool_meta(path: str | Path, *, model_id: str, pool: str,
                    default_reporting_scorer: str, gen_config: dict,
                    runs: list[dict], df=None) -> Path:
    """Write the ``<pool>.meta.json`` provenance sidecar next to a pool parquet.
    ``runs`` is a list of per-batch dicts (run_id, cohort, k, seed, n_rollouts, ...).
    The pool stores raw facts only; ``default_reporting_scorer`` records which named
    scorer reproduces the headline accuracy (``answer-match``)."""
    path = Path(path)
    meta = {
        "model_id": model_id, "pool": pool, "schema": "POOL_SCHEMA",
        "default_reporting_scorer": default_reporting_scorer,
        "gen_config": gen_config, "runs": runs,
    }
    if df is not None:
        meta["n_rollouts"] = int(len(df))
        meta["n_problems"] = int(df["unique_id"].nunique())
        if "dup_index" in df.columns:
            meta["n_distinct_completions"] = int((df["dup_index"] == 0).sum())
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
