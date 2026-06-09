<!-- code -->
```python
from __future__ import annotations
```

<!-- md -->
# Generating Rollout Pools — Natural Sampling on a GPU

This notebook produces the **naturally-sampled rollout pools** the rest of the
pipeline builds on: for each MATH problem it samples K completions straight from the
prompt (the model picks its own first token — no forced opener), scores them, and
writes a single self-contained `generations/<model-slug>/<pool>.parquet` to the
[`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)
dataset.

A **pool** is just *scored natural rollouts in the canonical schema*
(`schema.POOL_SCHEMA` = `ROLLOUTS_SCHEMA` + `is_correct` + `scorer_id`), with
per-batch provenance in a `<pool>.meta.json` sidecar.

Two examples:
1. **Generate K=64 rollouts for MATH-500.**
2. **Extend an existing pool** — top every problem up to at least K=64 (our case:
   the Oat-Zero `math500_passK` pool, where many problems only have 16).

The heavy lifting lives in the package — `generate.natural.generate_natural` (vLLM)
and the `data.pools` helpers (score / assemble / deficit / extend); this notebook
just wires up secrets, installs the code on a GPU, and uploads.

<!-- md -->
Check for Colab vs. script.

<!-- code -->
```python
try:
    from google.colab import userdata
    from IPython import get_ipython
    is_colab = get_ipython() is not None
except ImportError:
    is_colab = False
```

<!-- md -->
# Secrets

- `HF_TOKEN` — Hugging Face token with **write** access (to push pools).
- `HF_USERNAME` — your HF username; the dataset is `<HF_USERNAME>/math-rollouts`.
- `GITHUB_TOKEN` — only if the `math-rollouts` code repo is private.

<!-- code -->
```python
import os

if is_colab:
    HF_TOKEN = userdata.get("HF_TOKEN")
    HF_USERNAME = userdata.get("HF_USERNAME")
    try:
        GH_TOKEN = userdata.get("GITHUB_TOKEN")
    except Exception:
        GH_TOKEN = None
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_USERNAME = os.environ.get("HF_USERNAME", "ChrisMcCormick")
    GH_TOKEN = os.environ.get("GITHUB_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
assert HF_USERNAME, "set HF_USERNAME (Colab userdata or env var)"
print("HF user:", HF_USERNAME, "| HF token:", "set" if HF_TOKEN else "MISSING")
```

<!-- md -->
# Install the `math-rollouts` package (with the `[gen]` extra)

Generation needs vLLM, so unlike the compute/analysis notebooks we install the
**`[gen]`** extra (torch / transformers / vllm). On a fresh Colab GPU runtime the
vLLM install takes a few minutes.

<!-- code -->
```python
GH_OWNER = "chrisjmccormick"   # GitHub owner of the math-rollouts CODE repo
_auth = f"{GH_TOKEN}@" if GH_TOKEN else ""
pip_url = f"math_rollouts[gen] @ git+https://{_auth}github.com/{GH_OWNER}/math-rollouts.git"
!pip install -q "{pip_url}"
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Example 1 — K=64 rollouts for MATH-500

<!-- md -->
## Configure

- `MODEL_ID` — any registered model (non-thinking here; thinking models work too,
  with a larger `max_tokens` and the `post-think-v1` scorer).
- `POOL` — output pool name → `generations/<slug>/<POOL>.parquet`.
- `K` — rollouts per problem.
- `SEED` — vLLM sampling seed (batch identity; stored on the rows).

<!-- code -->
```python
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
POOL     = "math500_K64"
K        = 64
SEED     = 0
OUT_ROOT = "/content/math-rollouts-data"
```

<!-- md -->
## Generate, score, assemble

`generate_natural` loads the problems' prompts via the model's adapter, samples K
completions each on the GPU, and returns raw rollout rows. `pools.build_pool` then
scores them (the model's canonical scorer — `boxed-match-stop-v1` here) and conforms
them to `POOL_SCHEMA`.

<!-- code -->
```python
from math_rollouts.data.problems import load_problems_by_split
from math_rollouts.generate.natural import generate_natural
from math_rollouts.data import pools

problems = load_problems_by_split("math500")
rows = generate_natural(MODEL_ID, problems, k=K, run_id=0, seed=SEED)

pool_df, scorer_id = pools.build_pool(rows, model_id=MODEL_ID)
acc = pool_df.is_correct.mean()
print(f"{len(pool_df):,} rollouts over {pool_df.unique_id.nunique()} problems "
      f"| scorer {scorer_id} | accuracy {acc*100:.1f}%")
```

<!-- md -->
## Write + upload

Writes `<POOL>.parquet` (canonical schema) plus a `<POOL>.meta.json` provenance
sidecar, then uploads both to `generations/<slug>/` in the dataset.

<!-- code -->
```python
from pathlib import Path
from huggingface_hub import HfApi
from math_rollouts.config import GenConfig
from math_rollouts.data.hf import model_slug

slug = model_slug(MODEL_ID)
out = Path(OUT_ROOT) / "generations" / slug / f"{POOL}.parquet"
pools.write_pool(pool_df, out)
pools.write_pool_meta(
    out.with_suffix(".meta.json"), model_id=MODEL_ID, pool=POOL, scorer_id=scorer_id,
    gen_config=GenConfig().as_dict(),
    runs=[{"run_id": 0, "cohort": POOL, "k": K, "seed": SEED, "n_rollouts": len(pool_df)}],
    df=pool_df,
)

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/math-rollouts"
for f in (out, out.with_suffix(".meta.json")):
    api.upload_file(path_or_fileobj=str(f), repo_id=repo_id, repo_type="dataset",
                    path_in_repo=f"generations/{slug}/{f.name}",
                    commit_message=f"Add pool {slug}/{f.name} ({len(pool_df)} rollouts)")
print("uploaded", f"generations/{slug}/{POOL}.parquet")
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Example 2 — Extend a corpus to ≥ K per problem

The Oat-Zero `math500_passK` pool is pass@K: easy problems have as few as 16
rollouts while hard ones have hundreds. To guarantee **every** problem has at least
K=64 (so an even-K analysis can keep all 500), we generate only the per-problem
*deficit* and append it under a fresh `run_id`.

`pool_deficit` computes `{unique_id: target_k - have}` for the short problems;
`generate_natural` accepts that dict directly (per-problem sample counts via vLLM's
per-prompt sampling params). `ensure_pool_schema` migrates the pool to the canonical
schema first if it hasn't been already.

<!-- code -->
```python
OAT_MODEL = "sail/Qwen2.5-Math-1.5B-Oat-Zero"
OAT_POOL  = "math500_passK"
TARGET_K  = 64
SEED2     = 64
```

<!-- code -->
```python
from math_rollouts.data.hf import load_generation_parquet
from math_rollouts.data.problems import load_problems_by_ids

existing = pools.ensure_pool_schema(load_generation_parquet(OAT_MODEL, OAT_POOL), OAT_MODEL)
deficit = pools.pool_deficit(existing, TARGET_K)
print(f"{len(deficit)}/{existing.unique_id.nunique()} problems below K={TARGET_K}; "
      f"generating {sum(deficit.values()):,} new rollouts")

probs = load_problems_by_ids(list(deficit))
run_id = pools.next_run_id(existing)              # fresh batch id (no sample_idx collisions)
new_rows = generate_natural(OAT_MODEL, probs, k=deficit, run_id=run_id, seed=SEED2)
new_df, scorer_id = pools.build_pool(new_rows, model_id=OAT_MODEL)

combined = pools.extend_pool(existing, new_df)
print(f"pool: {len(existing):,} -> {len(combined):,} rollouts")
print("min rollouts/problem now:", combined.groupby("unique_id").size().min())
```

<!-- md -->
## Write + upload the extended pool

<!-- code -->
```python
slug = model_slug(OAT_MODEL)
out = Path(OUT_ROOT) / "generations" / slug / f"{OAT_POOL}.parquet"
pools.write_pool(combined, out)
pools.write_pool_meta(
    out.with_suffix(".meta.json"), model_id=OAT_MODEL, pool=OAT_POOL, scorer_id=scorer_id,
    gen_config=GenConfig().as_dict(),
    runs=[{"run_id": int(r), "n_rollouts": int((combined.run_id == r).sum())}
          for r in sorted(combined.run_id.unique())],
    df=combined,
)
api.upload_file(path_or_fileobj=str(out), repo_id=repo_id, repo_type="dataset",
                path_in_repo=f"generations/{slug}/{OAT_POOL}.parquet",
                commit_message=f"Extend {slug}/{OAT_POOL} to >=K={TARGET_K} ({len(combined)} rollouts)")
api.upload_file(path_or_fileobj=str(out.with_suffix(".meta.json")), repo_id=repo_id,
                repo_type="dataset", path_in_repo=f"generations/{slug}/{OAT_POOL}.meta.json",
                commit_message=f"Update {OAT_POOL} meta")
print("uploaded extended", f"generations/{slug}/{OAT_POOL}.parquet")
```

<!-- md -->
Re-run **`02 - Compute Nuclei for Rollouts`** on the extended pool to refresh its
per-token nucleus store, then **`03 - Analyze Rollout Nuclei`** to re-read the stats.

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
## Notes — thinking models & truncation (forward-looking)

- **Thinking models** (e.g. Qwen3) use the *same* `POOL_SCHEMA`. "Did `</think>`
  close?" and post-`</think>` correctness are handled by the **scorer**
  (`post-think-v1`, selected automatically for thinking adapters) — not by extra
  schema columns. Give them a much larger `max_tokens`.
- **Truncation / budget.** `finish_reason` (`stop` = self-stopped vs `length` =
  hit the budget) and the per-row `max_gen_len` record how each rollout ended.
  Because they're per-row, a pool can mix budgets — which sets up a future
  **length-extend**: re-generate the `finish_reason == "length"` rows at a higher
  cap and update them in place by `ROLLOUT_KEY` (`model_id, unique_id, run_id,
  branch_path, sample_idx`), bumping `max_gen_len`/`finish_reason`. (This notebook
  implements the *width*-extend — more rollouts per problem — above.)
```
