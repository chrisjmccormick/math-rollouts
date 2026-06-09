# Migrate three legacy guided-rollouts datasets into the new math-rollouts schema

## Context

The `math-rollouts` project has a new raw schema + versioned scoring system, and another
agent is currently rewriting the existing HF `math-rollouts` pools onto it (working in a
separate git worktree, now operating on the actual data parquets). We want to additionally
migrate three older rollout datasets generated during the `guided-rollouts` project, already
downloaded locally under `/home/ubuntu/guided-rollouts/math-random/generations/`:

- **deepseek-r1-distill-qwen-1.5b** — 5 **natural** pools (~102k+ rows; thinking model; mixed
  `max_gen_len` 5000/16384; some math12k-keyed, one math500-keyed).
- **qwen3-8b** — `math500_natural` (natural) + `math500_forced_token0`, `math500_forced_openers`
  (forced) + `trees/` (depth-4 nucleus trees); thinking model.
- **qwen3-8b-base** — `math500_forced_token0` (66k-row full first-token forced sweep); base model.

These feed two goals: (a) reusable scored pools in the new schema, and (b) **forced-opener tree
data for ongoing/expanding branch experiments and treeviz** (node model-probabilities + observed
accuracies; key finding: incorrect rollouts cluster in underperforming branches a few layers deep).

Two validated corrections to earlier assumptions: **deepseek completions DO contain literal
`</think>`** (so `post-think-v1` works; no special scorer needed) and **`run_id` is already
integer everywhere** (no string→int remap). Scope = all three dirs, executed in **two phases**.

## Target schema (FINALIZED — other agent's redesign, committed `a66e627`)

`src/math_rollouts/schema.py` (redesigned — the pool bakes NO verdict):
- `ROLLOUTS_SCHEMA`: model_id, unique_id `<split>/<subj>/<n>`, subject, answer, depth,
  branch_path, opener_token_ids, run_id, gen_config_id, seed, temperature, top_p, max_gen_len,
  sample_idx, completion_token_ids [EOS **incl** in the ids], completion_text,
  **finish_reason, stop_reason, terminal** (derived: emitted_eos/stop_string/truncated/repetition/…),
  **prompt_num_tokens, completion_num_tokens (EOS-EXCLUDED), total_num_tokens** — `num_tokens` is
  gone (renamed + EOS-excluded).
- `POOL_SCHEMA = ROLLOUTS_SCHEMA + answer_matches, has_boxed, answer_char_pos, answer_token_frac,
  dup_index`. **`is_correct` and `scorer_id` are DROPPED** — the pool stores criterion-free facts.
- `SCORES_SCHEMA` (per rollout×scorer): …, scorer_id, **verdict (correct|incorrect|unresolved)**,
  answer_matches, has_boxed, answer_char_pos, answer_token_frac, leak_class.
- `NUCLEI_SCHEMA` (per opener/leaf, unchanged): fork_token_id, nuc_prob, path_prob, branch_size,
  terminal, is_thinking.

Scorers (`score/scorers.py`): `answer-match` (default), `boxed-match`, `benchmark@budget=B`
(unresolved+strict), `leak-filtered@keep_frac`, `post-think-v1`.
`data.pools.default_scorer_id(model_id)` → `post-think-v1` for thinking models else `answer-match`.

> **Sequencing:** the other agent's schema is FINALIZED and committed (`a66e627`); adopt these
> names/columns directly (no v1 `is_correct`/`boxed-match-stop-v1`). Build + run Phase 1 against
> their `data/pools.py` — note the NEW signatures: `migrate_legacy_pool`/`build_pool`/
> `row_attributes` take `tok`/`eos_id`/`prompt_len` — and the parallel-migrator pattern in
> `/home/ubuntu/migrate_pools_par.py`, NOT the old `scripts/migrate_pools.py`.

> ⚠ **CORE PREREQUISITE — think-aware `answer_matches` (review resolution A).** The finalized
> analysis stack (difficulty/bandtable/nuclei_stats/policies/token_nuclei) reads the pool's stored
> `answer_matches` DIRECTLY as the reporting verdict, but `row_attributes` computes it
> FULL-completion for every model (no `is_thinking` branch) — the leaky reading `post-think-v1`
> exists to avoid. **Fix in the shared core (`data.pools.row_attributes`):** thread the adapter so
> `answer_matches` AND the placement facts (`answer_char_pos`/`answer_token_frac`, used by
> `leak-filtered`) are computed on the **post-`</think>` region** for thinking models — i.e.
> `answer_matches` == the model's default verdict (`check_correct_post_think` for thinking,
> full-completion `check_correct` otherwise). The already-converted non-thinking pools (Oat, base)
> are unaffected. **All 6 Phase-1 pools are thinking models, so this is a hard prerequisite** —
> coordinate the fix into the other agent's core so there's one source of truth.

---

## Phase 1 — Natural pools (low risk, reuses the redesigned migrator)

Covers **all 5 deepseek pools + qwen3-8b `math500_natural`** — all NATURAL (K-per-problem) → the
flat `POOL` path. **All six are thinking models**, so the CORE think-aware `answer_matches` fix
(above) is a hard prerequisite; with it in place the per-pool flow is a straight reuse of
`data/pools.py`.

**Prerequisite 1 — the core think-aware fix** (above): without it these pools are mis-reported
(full-completion correctness instead of post-`</think>`).

**Prerequisite 2 — register a deepseek adapter.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` is
not in `adapters/registry.py`. Add `DeepseekR1DistillAdapter` (sibling of `qwen3_think.py`):
`is_thinking=True`, a `prompt_ids` matching the deepseek chat template (**load-bearing for
`prompt_num_tokens`**), and `THINK_CLOSE` token id **151649** (NOT Qwen3's 151668) for
`terminal_ids` — Phase-1 scoring is string-based (`</think>`), so the id only matters for later
branch work, but the prompt template must be correct or `prompt_num_tokens` is wrong. qwen3-8b's
adapter already exists.

**Prerequisite 3 — per-model tokenizer + eos_id.** deepseek and qwen3-8b have DIFFERENT
tokenizers/EOS than the base Qwen that `migrate_pools_par.py` currently hard-codes. Generalize it
to take the source model's tokenizer + `eos_id` (used for `prompt_num_tokens`, EOS-excluded
`completion_num_tokens`, and `answer_token_frac`). **Verify per model** whether the legacy
`completion_token_ids` actually include a trailing EOS — if they don't, exclusion is a harmless
no-op, but note those lengths are already content-length.

**Per-pool steps** (reuse `data.pools.migrate_legacy_pool`/`build_pool` — NEW signatures take
`tok`/`eos_id`/`prompt_len` — plus `add_dup_index`/`write_pool`/`write_pool_meta`/`pool_drift_report`):
1. Project legacy df → `POOL_SCHEMA`; set `depth=0, branch_path=[], opener_token_ids=[]`; drop
   baggage (problem_idx, producer, initial_num_tokens, closed_think/n_inside_think/n_post_think,
   outcome_class, timestamp, level). Raw attrs (`answer_matches`, `has_boxed`, placement,
   `terminal`, lengths) are computed by `row_attributes`. `stop_reason` is absent in legacy → null
   → `terminal` derives `emitted_eos` for every `stop` row (any legacy stop-strings are
   indistinguishable — acceptable). Preserve cohort identity in `.meta.json` `runs[]`.
2. **Remap `unique_id`** via `scripts/migrate_unique_id_splits.py:build_maps` (needs
   `problems/math_problems.parquet` + `math500.parquet`, snapshot-downloaded; verify it still loads
   against the current tables): qwen3-8b `test/<subj>/<n>.json` → `math500/<subj>/<n>`; deepseek
   `train/<subj>/<n>` → keep `train/…` or flip to `math500/…` for holdout pools (e.g.
   `math500_cp_baseline`) per the maps.
3. **`answer_matches` is the post-`</think>` match** (via the core fix — these are thinking
   models). Emit `pool_drift_report`: it compares legacy full-completion `is_correct` vs the new
   post-think `answer_matches`, so **expect flips on truncated (`finish_reason==length`, no
   `</think>`) and think-block-only-boxed rows** — the intended, meaningful drift. **Review before
   upload.** `default_reporting_scorer` in the meta = `post-think-v1`.
4. `add_dup_index` (token-id identity; **keep all rollouts — NO dedup**; thinking models at T>0
   should have few natural repeats, but compute it). Write `generations/<slug>/<pool>.parquet` +
   `.meta.json` (`write_pool_meta(..., default_reporting_scorer="post-think-v1", ...)`) +
   `.drift.json`. Keep the **5 deepseek files as separate pool files** (all `run_id==0` → never
   union; ROLLOUT_KEY collisions). No `_token_nuclei` shards exist → `refresh_shard_answer_matches`
   is N/A.

**Phase-1 open item — null `completion_token_ids`** on 2 deepseek files
(`competition_math_random_labeled`, `math500_cp_baseline_64x16k`): **accept null** →
`completion_num_tokens`/`total_num_tokens`/`answer_token_frac` become null for those rows. Text
scorers (`answer-match`, `post-think-v1`) don't need ids; `benchmark@budget` keys off
`terminal`+`max_gen_len` (not length); `leak-filtered` falls back to a char-fraction. Re-tokenizing
risks EOS-accounting mismatch — only do it if token-level analysis later needs the ids.

---

## Phase 2 — Forced / nucleus datasets (new tooling, full-fidelity F1)

Covers **qwen3-8b `forced_token0` + `forced_openers` + `trees/`, and qwen3-8b-base `forced_token0`**.
Per your direction these go to the **full experiment layout** (`generations/<slug>/<exp>/
{nuclei,rollouts,scores}.parquet` + `manifest.json`), reconstructing branch structure so the trees
drive treeviz (node `nuc_prob`/`path_prob` + observed accuracies).

**New script `scripts/migrate_forced_experiment.py`** emitting `ROLLOUTS_SCHEMA` rollouts +
`NUCLEI_SCHEMA` nuclei + scored `SCORES_SCHEMA`:

- **Depth-1 sweeps (qwen3-8b-base 66k, qwen3-8b forced_token0) — cheap, do first.**
  `depth=1`, `opener_token_ids=[token_id]`, `branch_path=[child_idx]` where `child_idx` =
  rank of `token_id` by descending `nuc_prob` within the problem (recoverable from the file's own
  `(token_id, nuc_prob)` or `_provenance/` first-token nuclei). `nuclei` rows straight from
  `(unique_id, token_id, nuc_prob)`; `branch_size` = nucleus width; `terminal` from the adapter.
  Validated: `token_id == completion_token_ids[0]` for 100% of base rows.

- **Depth-3/4 `forced_openers` (1232 rows) — needs a tree hydrator (the hard part).**
  `trees/openers.parquet` already supplies ~90% of `NUCLEI_SCHEMA` (`leaf_depth`→depth,
  `opener_token_ids`, `path_prob`, `branch_size`); the missing `branch_path`, `fork_token_id`
  (=`opener_token_ids[-1]`), and leaf `nuc_prob` (=`inbound_prob`) come from `trees/trees/*.json`.
  Build a **`trees/*.json` → anytree `Node` hydrator** so `nucleus/openers.leaf_openers()` can be
  reused. ⚠ **Design discussion needed:** the JSON nodes are **multi-token segments**, not the
  one-token-per-node model in `nucleus/tree.py`. Decide `branch_path` granularity (per-segment vs
  expand to per-token) — this affects treeviz alignment with *future* freshly-generated depth-N
  experiments. Flagged with you as "discuss more."

- **Scoring:** qwen3-8b forced → `post-think-v1`; qwen3-8b-base → `boxed-match-stop-v1` (or the
  other agent's `answer-match`). qwen3-8b-base already carries `answer_frac`/`leak_class`/`keep_frac`
  → map directly onto a `leak-filter-v1` `scores.parquet` (or recompute via `analysis/positional.py`
  for reproducibility; `answer_char_pos` is not stored and would be recomputed).

- **Treeviz preservation:** carry the per-problem tree (probabilities) into the new layout — either
  keep `trees/trees/*.json` alongside the experiment or serialize the hydrated tree to whatever tree
  store the new system expects. Confirm the new repo's tree/treeviz representation during Phase 2.

## Clean up of uniform rollout

(Another agent's recommendation):

Now let me investigate #2 — the `math500_uniform_k16_d1` experiment that wasn't updated. Let me inspect its actual files and schema.

This is a full forced-opener **experiment** dir (rollouts + nuclei + scores + policies + manifest), for both non-thinking models — the depth-1 first-token nucleus sweep. Let me inspect the actual schemas to assess what an update would entail.

The `math500_uniform_k16_d1` experiment: moderately simple, well-contained

It's a full forced-opener experiment dir (rollouts + nuclei + scores + policies + manifest), for both **non-thinking** models, **depth-1**. Schema check vs the redesign:

| file | rows | status |
|---|--:|---|
| `nuclei.parquet` | 3,275 | **already conformant** — redesign didn't touch `NUCLEI_SCHEMA`; leave as-is |
| `rollouts.parquet` | 52,400 | stale: has old `num_tokens`+`finish_reason`; **missing** `stop_reason`/`terminal`/`prompt`/`completion`(EOS-excl)/`total_num_tokens` |
| `scores.parquet` | 52,400 | stale: old `is_correct`; **missing** `verdict`/`answer_matches`/`has_boxed` |

So it's **not a one-line rename, but not the hard case either** — it needs:
1. **rollouts** → add the 4 termination/length columns (reuse the same length/`terminal` logic the natural migrator uses; needs the Qwen tokenizer + prompt text). The branch columns (`depth`/`branch_path`/`opener_token_ids`) carry over **unchanged** — no tree hydration.
2. **scores** → regenerate through the new scorers via `score/run.py:score_file` (default `answer-match` — these are non-thinking, so the think-aware fix is irrelevant here) → `verdict`/`answer_matches`/`has_boxed`.
3. **policies.csv** regenerated from the new scores; **manifest.json** provenance bumped.

The only "careful" aspect is that it's a **different code path** than the natural-pool migrator (which deliberately skips forced files) — it's effectively the **depth-1 case of Phase 2's `migrate_forced_experiment.py`**. Because it's depth-1 + non-thinking, it sidesteps the genuinely hard Phase-2 work (tree hydration, per-segment vs per-token `branch_path`).

Worth knowing: the experiment is currently **incompatible with the merged analysis code** — `policies.py`/`load_scored_rollouts` expect `answer_matches`, but its scores still carry `is_correct` — so it can't be analyzed until updated.

**My recommendation:** fold it into Phase 2 as the depth-1 path's first target (it's the simplest forced case and a good way to validate that tooling), or I can do it now as a small one-off conform script since it needs no tree work. Want me to knock it out now, or leave it for the Phase-2 pass?


---

## Reuse vs new code

- **Reuse (NEW signatures):** `data/pools.py` (`migrate_legacy_pool`/`build_pool` now take
  `tok`/`eos_id`/`prompt_len`; `add_dup_index`, `write_pool`, `write_pool_meta(...,
  default_reporting_scorer=...)`, `pool_drift_report`, `refresh_shard_answer_matches`); the
  parallel-migrator pattern in `/home/ubuntu/migrate_pools_par.py` (generalize its hard-coded base
  tokenizer to per-model); `scripts/migrate_unique_id_splits.py` (build_maps + stream remap);
  `score/run.py:score_file` + `score/scorers.py:get_scorer`; `nucleus/openers.py:leaf_openers`;
  `data/hf.py` writers (`model_slug`, paths).
- **New / changed:** **core think-aware `answer_matches` in `data/pools.row_attributes`** (the
  prerequisite); `adapters/deepseek_distill.py` + registry; per-model tokenizer plumbing in the
  migrator; (Phase 2) `scripts/migrate_forced_experiment.py` + a `trees/*.json → anytree Node`
  hydrator; extend the natural-vs-forced predicate (forced files flagged by `gather_method`/
  `token_id`/`opener_idx`, not legacy `guided`/`branch_token_id`).

## Critical files

- `src/math_rollouts/data/pools.py` — **`row_attributes` think-aware fix (prerequisite)** + the
  reused migrator helpers
- `src/math_rollouts/adapters/registry.py` (+ new `deepseek_distill.py`)
- `/home/ubuntu/migrate_pools_par.py` (Phase 1 driver — generalize tokenizer per model);
  **new** `scripts/migrate_forced_experiment.py` (Phase 2)
- `src/math_rollouts/data/hf.py`, `src/math_rollouts/nucleus/openers.py` (+ tree hydrator),
  `src/math_rollouts/score/{run,scorers}.py`
- `src/math_rollouts/schema.py` — **FINALIZED (`a66e627`)**; adopt as-is

## Decisions

**Resolved:**
- **Think-aware `answer_matches`** (review resolution A) — `answer_matches` == the model's default
  verdict: post-`</think>` for thinking models, full-completion otherwise. Fix in the shared core
  before Phase 1. *(Confirmed with owner.)*
- **`subject` convention** — display string ("Counting & Probability"), matching the converted
  Oat/base pools.
- **Null token_ids** (2 deepseek files) — accept null (lengths null; text scorers unaffected).
- **Schema/scorer names** — finalized (`a66e627`); adopt directly.
- **No dedup** — keep all rollouts; `dup_index` flags natural repeats.

**Open (Phase 2):**
- **forced_openers `branch_path` granularity** (per-segment vs per-token) — ties to treeviz +
  continued depth-N experiments. (Flagged for discussion.)

## Verification

- **Schema conformance:** assert each output parquet's pyarrow schema equals `POOL_SCHEMA`
  (finalized columns incl. `answer_matches`/`has_boxed`/`terminal`/`stop_reason`/lengths/`dup_index`;
  NO `is_correct`/`scorer_id`); run `tests/test_schema.py`, `test_policies.py`, `test_adapters.py`
  (+ `test_tree_depth1.py` in Phase 2). Add an adapter test for deepseek `</think>` scoring.
- **Think-aware check:** on a thinking-model pool, confirm `answer_matches == check_correct_post_think`
  (not full-completion) on a spot sample — esp. a truncated row with no `</think>` (→ False) and a
  think-block-only-boxed row (→ False).
- **Drift review:** inspect `<pool>.drift.json` (post-think `answer_matches` vs legacy
  full-completion `is_correct`); truncated/think-only flips are expected and human-acknowledged.
- **EOS / lengths:** spot-check `completion_num_tokens` excludes a trailing EOS where present;
  `truncated` rows carry no EOS; null-token-id rows have null lengths.
- **Round-trip:** load via `data/hf.py:load_scored_rollouts` / `data/pools.py`; spot-check
  `answer_matches` + `unique_id` remap vs legacy rows.
- **No-merge guard:** verify the 5 deepseek pools remain distinct files (no ROLLOUT_KEY collisions).
- **Treeviz smoke test (Phase 2):** reconstruct one problem's tree from migrated `nuclei.parquet`
  + scores; confirm node probabilities + observed accuracies render.
