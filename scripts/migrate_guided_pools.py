#!/usr/bin/env python3
"""Phase 1: migrate the guided-rollouts NATURAL pools into ``POOL_SCHEMA``.

Source: the ``ChrisMcCormick/guided-rollouts`` HF dataset,
``math-random/generations/<slug>/<pool>.parquet`` — six natural (K-per-problem)
pools from two THINKING models:

    qwen3-8b/math500_natural                          (32 problems, K=16, 16k cap)
    deepseek-r1-distill-qwen-1.5b/competition_math_random_labeled   (12k problems)
    deepseek-r1-distill-qwen-1.5b/cp_random0_8
    deepseek-r1-distill-qwen-1.5b/nt_random100
    deepseek-r1-distill-qwen-1.5b/math500_cp_baseline_64x16k
    deepseek-r1-distill-qwen-1.5b/round2_baseline_K64_8k

Per pool: re-key ``unique_id`` to the split-aware ids (qwen3 is MATH-500-native
``test/<subj>/<n>.json``; deepseek is legacy math12k ``train/<subj>/<n>`` —
``build_maps`` covers both), re-derive the criterion-free raw attributes with the
THINK-AWARE core (``answer_matches`` == the post-``</think>`` verdict), and write
``generations/<slug>/<pool>.parquet`` + ``.meta.json`` + ``.drift.json`` for review.

Expectations baked in (from the source READMEs + the migration plan):
- both models are thinking models → ``default_reporting_scorer = post-think-v1``;
- the qwen3 legacy ``is_correct`` was ALREADY post-think → its drift should be ~0;
  deepseek's was full-completion → expect flips on truncated / think-only-boxed rows
  (the intended drift — review before upload);
- ``competition_math_random_labeled`` + ``math500_cp_baseline_64x16k`` have 100%
  NULL ``completion_token_ids`` → token-derived lengths/fracs are null (accepted),
  and ``dup_index`` falls back to completion-text identity;
- all deepseek pools are ``run_id == 0`` → they stay SEPARATE pool files (a union
  would collide on ``ROLLOUT_KEY``); cohort identity is recorded in ``runs[]``.

Scoring needs a real POSIX environment (``math_verify`` is broken on Windows) —
run this on Linux. The big deepseek file expands to several GB in memory; use a
box with >=16 GB RAM or migrate it alone via ``--pools``. Nothing is uploaded;
after reviewing the drift reports::

    python scripts/migrate_guided_pools.py --out-root /tmp/guided-migrated
    cat /tmp/guided-migrated/generations/*/*.drift.json
    hf upload ChrisMcCormick/math-rollouts /tmp/guided-migrated . \\
        --repo-type dataset --exclude "*.drift.json"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from math_rollouts.adapters import get_adapter
from math_rollouts.data.hf import load_problems_parquet, model_slug
from math_rollouts.data.pools import (
    default_scorer_id, migrate_legacy_pool, pool_drift_report, write_pool,
    write_pool_meta,
)

from migrate_pools import _is_natural_pool          # same dir; value-aware predicate
from migrate_unique_id_splits import build_maps

GUIDED_REPO = "ChrisMcCormick/guided-rollouts"
GEN_PREFIX = "math-random/generations"

# (model_id, pool, {run_id: cohort}) — cohort legends from the source READMEs.
PHASE1 = [
    ("Qwen/Qwen3-8B", "math500_natural", {
        0: "imp_screen (genuine-error screen over the base-Impossible band, 25 problems)",
        1: "xband_screen (cross-band candidates, 7 problems)",
    }),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "competition_math_random_labeled", {
        0: "full-pool K=8 labeling (gc 200) + extension/rescue rows to <=64 (gc 102, 16k cap)",
    }),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "cp_random0_8", {
        0: "Counting&Prob hard cases (0/8 correct), K=64 deep re-sample @16k",
    }),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "nt_random100", {
        0: "Number-Theory hard cases (0-correct), K=64 deep re-sample @16k",
    }),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "math500_cp_baseline_64x16k", {
        0: "C&P MATH-500-holdout baseline vs FT'd models, K=64 @16k",
    }),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "round2_baseline_K64_8k", {
        0: "round-2 set baseline vs teacher-KV, 64 x 25 steps @8k (sample_idx 0..1599)",
    }),
]


def _fetch(in_root: Path | None, slug: str, pool: str) -> Path:
    """Local file under ``--in-root`` if given, else download just that parquet."""
    rel = f"{GEN_PREFIX}/{slug}/{pool}.parquet"
    if in_root is not None:
        p = in_root / rel
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(GUIDED_REPO, repo_type="dataset", filename=rel))


def _remap_ids(legacy: pd.DataFrame, cmap: dict, pool: str) -> pd.DataFrame:
    new = legacy["unique_id"].map(cmap)
    if new.isna().any():
        missing = sorted(legacy.loc[new.isna(), "unique_id"].unique())
        raise SystemExit(f"{pool}: {len(missing)} unmapped unique_ids "
                         f"(first: {missing[:5]}) — refusing to write a partial remap")
    return legacy.assign(unique_id=new)


def _prompt_len_map(model_id: str, uids, tok, text_map: dict) -> dict:
    """unique_id -> prompt_num_tokens under the model's adapter (load-bearing)."""
    adapter = get_adapter(model_id)
    missing = [u for u in uids if u not in text_map]
    if missing:
        raise SystemExit(f"{len(missing)} unique_ids have no problem text "
                         f"(first: {missing[:5]}) — prompt_num_tokens would be wrong")
    return {u: len(adapter.prompt_ids({"problem": text_map[u]}, tok)) for u in uids}


def _gen_config_observed(legacy: pd.DataFrame) -> dict:
    """The sampling config as OBSERVED on the legacy rows (the per-row columns are
    authoritative; this is provenance, incl. mixed budgets within a cohort)."""
    cfg = {
        "temperature": sorted(float(t) for t in legacy["temperature"].unique()),
        "top_p": sorted(float(t) for t in legacy["top_p"].unique()),
        "max_tokens": sorted(int(t) for t in legacy["max_gen_len"].unique()),
        "gen_config_ids": sorted(int(g) for g in legacy["gen_config_id"].unique()),
        "source": f"{GUIDED_REPO} (observed per-row values)",
    }
    if "top_k" in legacy.columns:
        cfg["top_k"] = sorted(int(t) for t in legacy["top_k"].dropna().unique())
    return cfg


def _runs_meta(migrated: pd.DataFrame, legend: dict) -> list[dict]:
    out = []
    for rid, g in migrated.groupby("run_id"):
        out.append({"run_id": int(rid), "cohort": legend.get(int(rid)),
                    "n_rollouts": int(len(g)),
                    "n_problems": int(g["unique_id"].nunique()),
                    "n_distinct_completions": int((g["dup_index"] == 0).sum())})
    return out


def migrate_one(model_id: str, pool: str, legend: dict, *, in_root: Path | None,
                out_root: Path, cmap: dict, text_map: dict, tok) -> None:
    slug = model_slug(model_id)
    src = _fetch(in_root, slug, pool)
    legacy = pd.read_parquet(src)
    if not _is_natural_pool(legacy):
        raise SystemExit(f"{slug}/{pool}: does not look like a NATURAL pool — "
                         f"forced files are Phase 2, refusing")
    legacy = _remap_ids(legacy, cmap, pool)

    null_ids = int(legacy["completion_token_ids"].isna().sum())
    plen = _prompt_len_map(model_id, legacy["unique_id"].unique().tolist(), tok, text_map)
    migrated = migrate_legacy_pool(legacy, model_id=model_id, tok=tok,
                                   prompt_len=plen, eos_id=tok.eos_token_id)
    drift = pool_drift_report(legacy, migrated)
    drift["null_completion_token_ids"] = null_ids
    print(f"  {slug}/{pool}: {drift['n_rollouts']:,} rollouts "
          f"({null_ids:,} null-token-id) | post-think vs legacy flips "
          f"{drift['n_flips']:,} (+{drift['flip_to_correct']}/-{drift['flip_to_incorrect']}) "
          f"| {drift['problems_band_moved']}/{drift['n_problems']} problems changed band")

    dst = out_root / "generations" / slug / f"{pool}.parquet"
    write_pool(migrated, dst)
    write_pool_meta(dst.with_suffix(".meta.json"), model_id=model_id, pool=pool,
                    default_reporting_scorer=default_scorer_id(model_id),
                    gen_config=_gen_config_observed(legacy),
                    runs=_runs_meta(migrated, legend), df=migrated)
    (out_root / "generations" / slug / f"{pool}.drift.json").write_text(
        json.dumps(drift, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--in-root", type=Path, default=None,
                    help="local copy of the guided-rollouts dataset "
                         "(contains math-random/generations/...); else download")
    ap.add_argument("--pools", nargs="*", default=None,
                    help="subset of pool names to migrate (default: all six)")
    a = ap.parse_args()

    if os.name == "nt":
        print("WARNING: math_verify needs POSIX — answer_matches will be all-False "
              "garbage on Windows. Mechanics-smoke only; run the real migration on Linux.")

    todo = [(m, p, l) for m, p, l in PHASE1 if a.pools is None or p in a.pools]
    if a.pools and len(todo) != len(a.pools):
        known = {p for _, p, _ in PHASE1}
        raise SystemExit(f"unknown pool(s): {sorted(set(a.pools) - known)}")

    mp = load_problems_parquet("math_problems")
    cmap, _ = build_maps(mp, load_problems_parquet("math500"))
    text_map = dict(zip(mp.unique_id, mp.problem))
    print(f"id map: {len(cmap):,} source ids | problem texts: {len(text_map):,}")

    from transformers import AutoTokenizer
    a.out_root.mkdir(parents=True, exist_ok=True)
    toks: dict[str, object] = {}
    for model_id, pool, legend in todo:
        if model_id not in toks:
            toks[model_id] = AutoTokenizer.from_pretrained(model_id)
        print(f"{model_id}:")
        migrate_one(model_id, pool, legend, in_root=a.in_root, out_root=a.out_root,
                    cmap=cmap, text_map=text_map, tok=toks[model_id])
    print(f"\nDONE -> {a.out_root}  (review *.drift.json, then upload — "
          f"drift on deepseek pools is EXPECTED: post-think vs legacy full-completion)")


if __name__ == "__main__":
    main()
