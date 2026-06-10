#!/usr/bin/env python3
"""One-off: migrate the legacy natural-sampled pools to the canonical ``POOL_SCHEMA``.

The flat pools (``generations/<slug>/<pool>.parquet``) came from an unreleased dev
project and carry a 24-column legacy schema with baggage (``problem_idx=-1``,
``producer``, ``initial_num_tokens``, think-segmentation fields, per-row
``timestamp``, redundant ``level``). This rewrites each natural pool to
``schema.POOL_SCHEMA`` (= ``ROLLOUTS_SCHEMA`` + the criterion-free answer/match facts
+ ``dup_index``), **deriving** the raw attributes (``answer_matches`` and the rest)
from ``completion_text`` + the vLLM fields so they are reproducible from this repo's
code, and writes a ``<pool>.meta.json`` provenance sidecar. It also refreshes the
copied correctness (``is_correct`` -> ``answer_matches``) and attaches ``dup_index`` in
any matching ``<pool>_token_nuclei`` shards so pool and shards agree.

For a faster fan-out across cores (used for the actual re-derivation), see the
parallel runner; this serial version reuses the same library functions.

Forced-opener experiment files (``*_uniform_openers*`` — they carry ``guided`` /
branch columns) are NOT pools; they're skipped.

Writes only the changed files to ``--out-root`` for review; upload is a separate
step (mirrors ``migrate_unique_id_splits.py``)::

    python scripts/migrate_pools.py --out-root /path/to/migrated [--in-root SNAP]

then, after reviewing the drift report::

    hf upload ChrisMcCormick/math-rollouts /path/to/migrated . --repo-type dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from math_rollouts.adapters import get_adapter
from math_rollouts.config import GenConfig
from math_rollouts.data.hf import load_problems_parquet, model_slug
from math_rollouts.data.pools import (
    default_scorer_id, migrate_legacy_pool, pool_drift_report,
    refresh_shard_answer_matches, write_pool, write_pool_meta,
)

REPO = "ChrisMcCormick/math-rollouts"
DEFAULT_MODELS = ["Qwen/Qwen2.5-Math-1.5B", "sail/Qwen2.5-Math-1.5B-Oat-Zero"]
# run_id -> cohort legend, from the dev-project generation README (provenance only).
RUN_LEGEND = {
    0: "math12k_L4_5_K64", 1: "math12k_passK", 2: "math12k_additional",
    3: "math500_passK (canonical K=64)", 4: "math500_passK (pak K=256 extension)",
}


def _is_natural_pool(df) -> bool:
    """A flat natural pool, not a forced-opener experiment or a derived table.
    Forced files are flagged by the legacy dev-project columns (``branch_token_id``)
    or the guided-rollouts forced columns (``token_id`` / ``opener_idx``). The
    guided-rollouts ``guided``/``gather_method`` columns exist on NATURAL files too
    (all-False / 'natural'), so those are checked by VALUE, not presence."""
    cols = set(df.columns)
    if {"branch_token_id", "branch_pos", "token_id", "opener_idx"} & cols:
        return False
    if "guided" in cols and bool(df["guided"].any()):
        return False
    if "gather_method" in cols and not set(df["gather_method"].unique()) <= {"natural"}:
        return False
    return {"completion_token_ids", "is_correct", "unique_id"} <= cols


def _runs_meta(df) -> list[dict]:
    out = []
    for rid, g in df.groupby("run_id"):
        out.append({"run_id": int(rid), "cohort": RUN_LEGEND.get(int(rid)),
                    "n_rollouts": int(len(g)),
                    "n_problems": int(g["unique_id"].nunique()),
                    "n_distinct_completions": int((g["dup_index"] == 0).sum())
                    if "dup_index" in g else None})
    return out


def _problem_text_map():
    """unique_id -> problem text across train/* (math_problems) and math500/*."""
    m = {}
    for nm in ("math_problems", "math500"):
        d = load_problems_parquet(nm)
        m.update(dict(zip(d["unique_id"], d["problem"])))
    return m


def _prompt_len_map(model_id, uids, tok, text_map):
    """unique_id -> prompt_num_tokens (rendered prompt length under the adapter)."""
    adapter = get_adapter(model_id)
    out = {}
    for uid in uids:
        t = text_map.get(uid)
        if t is not None:
            out[uid] = len(adapter.prompt_ids({"problem": t}, tok))
    return out


def migrate_one(in_root: Path, out_root: Path, model_id: str, pool_path: Path,
                tok, text_map) -> None:
    pool = pool_path.stem
    slug = model_slug(model_id)
    legacy = pd.read_parquet(pool_path)
    if not _is_natural_pool(legacy):
        print(f"  skip {slug}/{pool}.parquet (not a natural pool)")
        return

    plen = _prompt_len_map(model_id, legacy["unique_id"].unique().tolist(), tok, text_map)
    migrated = migrate_legacy_pool(legacy, model_id=model_id, tok=tok,
                                   prompt_len=plen, eos_id=tok.eos_token_id)
    drift = pool_drift_report(legacy, migrated)
    print(f"  {slug}/{pool}: {drift['n_rollouts']:,} rollouts | answer-match flips "
          f"{drift['n_flips']:,} (+{drift['flip_to_correct']}/-{drift['flip_to_incorrect']}) "
          f"| {drift['problems_band_moved']}/{drift['n_problems']} problems changed band")

    dst = out_root / "generations" / slug / f"{pool}.parquet"
    write_pool(migrated, dst)
    write_pool_meta(dst.with_suffix(".meta.json"), model_id=model_id, pool=pool,
                    default_reporting_scorer=default_scorer_id(model_id),
                    gen_config=GenConfig().as_dict(),
                    runs=_runs_meta(migrated), df=migrated)
    (out_root / "generations" / slug / f"{pool}.drift.json").write_text(
        json.dumps(drift, indent=2), encoding="utf-8")

    # Refresh copied correctness (is_correct -> answer_matches) + attach dup_index in
    # matching token_nuclei shards, if present.
    shard_dir = in_root / "generations" / slug / f"{pool}_token_nuclei"
    if shard_dir.is_dir():
        n = 0
        for shard in sorted(shard_dir.glob("*.parquet")):
            sdf = refresh_shard_answer_matches(pd.read_parquet(shard), migrated)
            out_shard = out_root / "generations" / slug / f"{pool}_token_nuclei" / shard.name
            out_shard.parent.mkdir(parents=True, exist_ok=True)
            sdf.to_parquet(out_shard, compression="zstd")
            n += 1
        # copy the shard sidecars untouched
        for side in ("_meta.json",):
            src = shard_dir / side
            if src.exists():
                (out_root / "generations" / slug / f"{pool}_token_nuclei" / side).write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"      refreshed answer_matches + dup_index in {n} {pool}_token_nuclei shards")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--in-root", type=Path, default=None,
                    help="local dataset snapshot; if omitted, download the relevant files")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    a = ap.parse_args()

    in_root = a.in_root
    if in_root is None:
        from huggingface_hub import snapshot_download
        patterns = []
        for m in a.models:
            s = model_slug(m)
            patterns += [f"generations/{s}/*.parquet",
                         f"generations/{s}/*_token_nuclei/*.parquet",
                         f"generations/{s}/*_token_nuclei/_meta.json"]
        in_root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns))
        print(f"snapshot at {in_root}")

    from transformers import AutoTokenizer
    text_map = _problem_text_map()

    a.out_root.mkdir(parents=True, exist_ok=True)
    for model_id in a.models:
        slug = model_slug(model_id)
        gdir = in_root / "generations" / slug
        if not gdir.is_dir():
            print(f"{model_id}: no generations dir, skipping")
            continue
        print(f"{model_id}:")
        tok = AutoTokenizer.from_pretrained(model_id)
        for pool_path in sorted(gdir.glob("*.parquet")):
            migrate_one(in_root, a.out_root, model_id, pool_path, tok, text_map)
    print(f"\nDONE -> {a.out_root}  (review *.drift.json, then upload)")


if __name__ == "__main__":
    main()
