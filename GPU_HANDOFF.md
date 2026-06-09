# GPU handoff — migrate pools, extend Oat-Zero, recompute nuclei

You are picking up after a code change that (1) introduced a canonical pool schema
(`schema.POOL_SCHEMA` = `ROLLOUTS_SCHEMA` + `is_correct` + `scorer_id`) and (2) added
true natural-sampling generation. The code is merged to `main`
(`github.com/chrisjmccormick/math-rollouts`). Your job is the data-side rollout on a
GPU instance: migrate the existing pools to the new schema, top up the Oat-Zero
MATH-500 pool to ≥64 rollouts/problem, recompute its nucleus store, and re-verify the
analysis.

**Dataset:** `ChrisMcCormick/math-rollouts` (HF, `repo_type=dataset`).
**Models:** base `Qwen/Qwen2.5-Math-1.5B` (slug `qwen2.5-math-1.5b`), RL-tuned
`sail/Qwen2.5-Math-1.5B-Oat-Zero` (slug `qwen2.5-math-1.5b-oat-zero`).

Do the steps in order. **Stop and report to the human after Step 1's drift report**
before uploading anything — re-scoring can move correctness, and everything downstream
depends on it.

---

## Step 0 — Environment

Linux GPU instance (A100-class is plenty; Oat-Zero completions are short). Note:
`math_verify` (the scorer) needs a real POSIX environment — it works on Linux, and
this is why scoring couldn't be validated on the Windows dev box.

```bash
git clone https://github.com/chrisjmccormick/math-rollouts.git
cd math-rollouts
pip install -e '.[gen]'          # torch / transformers / vllm + the CPU deps

export HF_TOKEN=...              # WRITE-scoped token (uploads)
export HF_USERNAME=ChrisMcCormick
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

Sanity-check the package imports and the canonical schema:

```bash
python -c "from math_rollouts.data import pools; from math_rollouts.generate.natural import generate_natural; from math_rollouts.schema import POOL_SCHEMA; print(len(POOL_SCHEMA.names), 'pool cols')"
```

---

## Step 1 — Migrate the legacy pools to `POOL_SCHEMA` (CPU)

Re-scores `is_correct` canonically (`boxed-match-stop-v1`), drops the legacy baggage
columns, writes `<pool>.meta.json`, and refreshes the copied `is_correct` in any
`*_token_nuclei` shards. Writes a **reviewable copy** to `--out-root` (does NOT upload).

```bash
python scripts/migrate_pools.py --out-root /tmp/migrated
```

For each pool it prints a one-line drift summary and writes `<pool>.drift.json`. **Read
these and report to the human:**

```bash
cat /tmp/migrated/generations/*/*.drift.json
```

Each report has `n_flips` (rollouts whose `is_correct` changed vs the legacy values),
`flip_to_correct` / `flip_to_incorrect`, and `problems_band_moved` (problems that
changed difficulty band). A handful of flips is expected; **if `problems_band_moved`
is large (say >5% of problems), pause** — it means the published difficulty bands and
analysis numbers will shift, and the human should weigh in first.

### Upload the migrated files (after the human OKs the drift)

Upload everything under `/tmp/migrated` EXCEPT the `*.drift.json` review artifacts:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/tmp/migrated", path_in_repo=".",
    repo_id="ChrisMcCormick/math-rollouts", repo_type="dataset",
    ignore_patterns=["*.drift.json"],
    commit_message="Migrate pools to POOL_SCHEMA (re-scored is_correct)",
)
```

This overwrites the legacy pool parquets + their token_nuclei shards in place. After
this, base `math500_passK` is fully done (no GPU needed for the base model).

---

## Step 2 — Extend Oat-Zero `math500_passK` to ≥64 rollouts/problem (GPU)

The pass@K pool has as few as 16 rollouts on easy problems. Expect **448 / 500
problems short, +48 each → ~21,504 new rollouts** under a fresh `run_id` (=2; existing
run_ids are 0/1). Run this after Step 1 so the pool on the hub is already canonical.

```python
from pathlib import Path
from huggingface_hub import HfApi
from math_rollouts.config import GenConfig
from math_rollouts.data.hf import load_generation_parquet, model_slug
from math_rollouts.data.problems import load_problems_by_ids
from math_rollouts.generate.natural import generate_natural
from math_rollouts.data import pools

MODEL, POOL, TARGET_K, SEED = "sail/Qwen2.5-Math-1.5B-Oat-Zero", "math500_passK", 64, 64
OUT_ROOT = "/tmp/gen"

existing = pools.ensure_pool_schema(load_generation_parquet(MODEL, POOL), MODEL)
deficit = pools.pool_deficit(existing, TARGET_K)
print(len(deficit), "problems short;", sum(deficit.values()), "new rollouts")

probs = load_problems_by_ids(list(deficit))
run_id = pools.next_run_id(existing)
rows = generate_natural(MODEL, probs, k=deficit, run_id=run_id, seed=SEED)   # vLLM
new_df, scorer_id = pools.build_pool(rows, model_id=MODEL)                   # scores (CPU)

combined = pools.extend_pool(existing, new_df)
mn = combined.groupby("unique_id").size().min()
print(f"{len(existing):,} -> {len(combined):,} rollouts; min/problem now {mn}")
assert mn >= TARGET_K, "some problem still below target K"

slug = model_slug(MODEL)
out = Path(OUT_ROOT) / "generations" / slug / f"{POOL}.parquet"
pools.write_pool(combined, out)
pools.write_pool_meta(
    out.with_suffix(".meta.json"), model_id=MODEL, pool=POOL, scorer_id=scorer_id,
    gen_config=GenConfig().as_dict(),
    runs=[{"run_id": int(r), "n_rollouts": int((combined.run_id == r).sum())}
          for r in sorted(combined.run_id.unique())], df=combined)

api = HfApi()
for f in (out, out.with_suffix(".meta.json")):
    api.upload_file(path_or_fileobj=str(f), repo_id="ChrisMcCormick/math-rollouts",
                    repo_type="dataset", path_in_repo=f"generations/{slug}/{f.name}",
                    commit_message=f"Extend {slug}/{POOL} to >=K={TARGET_K}")
```

Report back: problems-short count, new-rollout count, and the final min rollouts/problem
(should be ≥64).

> This is the `01 - Generate Rollouts` notebook's "Example 2", run directly. **Example 1**
> in that notebook (generate a fresh `math500_K64` pool for the *base* model) is a
> demonstration — only run it if a new base-model K=64 pool is actually wanted.

---

## Step 3 — Recompute the Oat-Zero nucleus store on the extended pool (GPU)

The extend added rollouts, so the uploaded `math500_passK_token_nuclei` shards are now
stale (they only cover the old rollouts). Regenerate them from the extended pool.

```python
from huggingface_hub import HfApi
from math_rollouts.analysis.token_nuclei import build_token_nuclei
from math_rollouts.data.hf import model_slug

MODEL, POOL, OUT = "sail/Qwen2.5-Math-1.5B-Oat-Zero", "math500_passK", "/tmp/nuclei"
stats, paths = build_token_nuclei(MODEL, POOL, OUT, shard_size=1,
                                  logit_dtype="float16", device="cuda")
print(stats)   # headline counts only (full stats are computed in analysis notebook 03)

slug = model_slug(MODEL)
HfApi().upload_folder(folder_path=str(paths["dir"]), repo_type="dataset",
                      repo_id="ChrisMcCormick/math-rollouts",
                      path_in_repo=f"generations/{slug}/{paths['dir'].name}",
                      commit_message=f"Recompute {POOL} nucleus store (extended pool)")
```

This is the `02 - Compute Nuclei for Rollouts` notebook. Leave `top_k` at its default
(20) so it stays consistent with the base-model store. (Separate future task, not for
now: the human wants to eventually regenerate the nucleus stores with a *higher* `top_k`
to drop the "capped at 20" qualifier — if/when you do that, also pass a matching
`top_k=` to `nuclei_stats.summarize_nuclei` in the analysis, since its default is 20.)

---

## Step 4 — Verify the analysis (CPU)

Confirm the extend achieved its goal — every MATH-500 problem now supports an even
K=64 sample — and spot-check the recomputed stats:

```python
from math_rollouts.data.hf import load_token_nuclei_pool
from math_rollouts.analysis.nuclei_stats import even_k_sample, summarize_nuclei

df = load_token_nuclei_pool("sail/Qwen2.5-Math-1.5B-Oat-Zero", "math500_passK",
        columns=["unique_id","subject","sample_idx","is_correct","n_tokens",
                 "nuc_sizes","chosen_is_top1"])
bal = even_k_sample(df, 64, seed=0)
assert bal.unique_id.nunique() == 500, "even-K=64 should now keep all 500 problems"
print(summarize_nuclei(bal))
```

The full report (per-difficulty, position, correct/incorrect) lives in
`03 - Analyze Rollout Nuclei`; re-run it if the human wants the refreshed figures.

---

## Quick reference / gotchas

- **Order matters:** Step 1 (migrate) before Step 2 (extend) — extend appends to the
  canonical pool. `ensure_pool_schema` will migrate-on-load if you skip Step 1, but
  then the hub copy stays legacy.
- **Fresh `run_id` per batch.** `pools.next_run_id` handles this; `(unique_id, run_id,
  sample_idx)` stays unique so nothing collides.
- **Sampling is temperature + top-p only.** `top_k` is a nucleus-size cap, NOT a
  generation limiter — don't add it to `SamplingParams`.
- **Scorer:** `boxed-match-stop-v1` for these non-thinking models (auto-selected).
  Thinking models would use `post-think-v1` — same schema.
- **Precision:** nuclei are recomputed in bfloat16 to match vLLM generation; keep it.
- **Don't upload review artifacts** (`*.drift.json`) to the dataset.
- The notebooks under `notebooks/` are markdown-form; the snippets above run the same
  package functions directly, so you don't need the `.md`→`.ipynb` converter.
```
