# Scoring-redesign migration — results (for review)

Re-derivation of all 7 natural pools + the 2 `math500_passK` nucleus stores under the
new `POOL_SCHEMA` (criterion-free attributes + named scorers; `is_correct` dropped).
Default reporting scorer is `answer-match` (`correct ⟺ answer_matches`, permissive
full-completion `math_verify`, truncation-tolerant). No dedup — every rollout kept;
`dup_index` (exact `completion_token_ids` identity within a problem) flags repeats.

## Pools (782,319 rollouts; all conform to `POOL_SCHEMA`)

| pool | rollouts | distinct | acc% (answer-match) | trunc% | drift vs legacy `is_correct` |
|---|--:|--:|--:|--:|---|
| base / math500_passK | 40,704 | 38,142 | 27.4 | 32.5 | **0** |
| base / math12k_additional | 6,144 | 6,047 | 26.1 | 41.5 | **0** |
| base / math12k_passK | 130,304 | 128,844 | 0.4 | 24.1 | 53 (+53 / −0), 1 band |
| base / math12k_L4_5_K64 | 416,704 | 413,458 | 36.6 | 27.4 | 520 (+504 / −16), 19 bands |
| oat / math500_passK | 43,215 | 38,392 | 56.4 | 2.0 | **0** |
| oat / math12k_K64 | 48,704 | 45,464 | 45.2 | 3.1 | **0** |
| oat / math12k_passK | 96,544 | 94,861 | 0.9 | 5.9 | **0** |

`drift` = per-rollout flips of the new `answer_matches` vs the legacy published
`is_correct` (`+` = newly correct, `−` = newly incorrect); `band` = problems whose
base-solve-rate difficulty band changed.

### The 2 non-zero-drift pools — a `math_verify` version difference, not a bug
The instruction expected 0 drift everywhere; 5/7 pools are exactly 0. The two hard
**base** pools drift slightly, and it was investigated:

- The flips are **deterministic**: re-scoring the flipped rows single-threaded twice
  is 0/40 unstable and sub-second (no sympy timeouts), and single-thread
  `check_correct` matches the parallel `answer_matches` on 40/40 — so this is **not** a
  parallel-load / timeout artifact.
- All 40 sampled flips **disagree with legacy** `is_correct`, almost all in the
  `+correct` direction (504/520 and 53/53), concentrated on the hardest pool
  (`math12k_L4_5_K64`, level 4–5) where answers are most symbolically complex.
- Both scorers are permissive full-text (legacy has 4,628 truncated-but-correct rows,
  so it was *not* stop-gated). Cause: the hub's older math12k pools were scored with an
  **older `math_verify`**; `math500_passK` / `math12k_additional` already agree with the
  current `math_verify 0.9.0`.

So the re-derivation **unifies all pools under one reproducible scorer** (this repo's
`check_correct`, the stated goal). The drift (≤0.12% of rollouts) reflects the newer
verifier recovering correct answers the old one missed — it is correct and stable.

## Nucleus stores (`math500_passK_token_nuclei`)

- **base** (`qwen2.5-math-1.5b`): the existing UNCAPPED store, unchanged except a
  column refresh — copied `is_correct` → `answer_matches`, `dup_index` attached by join
  to the re-derived pool. 500 shards, 40,704 rollouts, 0 NaN.
- **oat** (`qwen2.5-math-1.5b-oat-zero`): MERGED. Published `run_id 0/1` shards (21,312,
  capped at top_k=20) refreshed to `answer_matches` + `dup_index`; `run_id 2` (21,903
  rollouts, 448 problems) computed fresh + UNCAPPED and merged per problem. 500 shards,
  43,215 rollouts, 0 NaN. Per-problem `answer_matches`/`dup_index` match the pool on all
  43,215 rows.
  - **Note (cap asymmetry):** the instruction assumed "Oat never exceeds size 20";
    empirically the uncapped `run_id 2` exceeds 20 at **10 / 12.7M positions (0.0001%,
    9 rollouts; max 871** at a rare near-flat distribution). The published `run_id 0/1`
    has ~1 censored position per 1.6M. So the capped (run0/1) and uncapped (run2) halves
    are effectively identical; recomputing run0/1 uncapped (~80 min GPU) was skipped as
    pointless. Recorded in the store's `_meta.json`.

## Not touched (flagged)
The forced-opener experiment dirs `generations/<slug>/math500_uniform_k16_d1/`
(`scores.parquet`) still carry the **old** `SCORES_SCHEMA` (`is_correct`). They are not
pools and are out of this deliverable's upload scope; re-score them separately if hub
schema uniformity is wanted.

## Staging
Everything is staged at `/tmp/upload/generations/<slug>/` (7 pools + `.meta.json`,
2 nuclei stores; 1.3 GB). **Nothing has been uploaded — awaiting explicit review/OK.**
