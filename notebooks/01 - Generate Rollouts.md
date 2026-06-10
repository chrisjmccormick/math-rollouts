<!-- code -->
```python
from __future__ import annotations
```

<!-- md -->
# Generating Rollout Pools — Natural Sampling on a GPU

This notebook produces the **naturally-sampled rollout pools** the rest of the
pipeline builds on: for each MATH problem it samples K completions straight from the
prompt (the model picks its own first token — no forced opener), computes the raw
answer/match facts, and writes a single self-contained
`generations/<model-slug>/<pool>.parquet` to the
[`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)
dataset.

A **pool** is just *natural rollouts in the canonical schema*
(`schema.POOL_SCHEMA` = `ROLLOUTS_SCHEMA` + the criterion-free answer/match facts +
`dup_index`), with per-batch provenance in a `<pool>.meta.json` sidecar. The pool
bakes **no verdict**: it stores the *facts* about each rollout (`answer_matches`,
`has_boxed`, answer placement, termination, lengths) and accuracy is reproduced by a
**named scorer** over those facts. The default reporting scorer is `answer-match`
(`correct ⟺ answer_matches`); for **thinking models** it is `post-think-v1` — and
`answer_matches` already equals that verdict, because the facts are computed on the
post-`</think>` region for thinking adapters.

Three examples:
1. **Generate K=64 rollouts for MATH-500** (non-thinking Qwen2.5-Math).
2. **Extend an existing pool** — top every problem up to at least K=64 (our case:
   the Oat-Zero `math500_passK` pool, where many problems only have 16).
3. **Thinking model: Qwen3-8B on MATH-500** — top up (or bootstrap) the
   `math500_natural` pool with a reasoning-sized token budget.

The heavy lifting lives in the package — `generate.natural.generate_natural` (vLLM)
and the `data.pools` helpers (facts / assemble / deficit / extend); this notebook
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
# Install the `math-rollouts` package + the generation stack

Generation needs vLLM, and **where it comes from depends on the box**:

- **Colab**: the runtime ships CUDA 12, but current vLLM releases default to
  CUDA-13 wheels and die at import with `libcudart.so.13: cannot open shared
  object file`. So we install a **pinned, Colab-proven stack first** (vLLM
  0.12.0 — the newest CUDA-12 line — which itself pins torch 2.9.0; mirrors
  `requirements/colab-gen.txt`), then the package *without* the `[gen]` extra
  so pip doesn't re-resolve vllm/torch. Ignore pip's "dependency conflicts"
  warnings about Colab-preinstalled packages (google-adk / gradio / cudf etc.)
  — they're unrelated.
- **Bare Linux GPU box**: the unpinned **`[gen]`** extra is fine.

Either way the vLLM install takes a few minutes on a fresh runtime.

<!-- code -->
```python
GH_OWNER = "chrisjmccormick"   # GitHub owner of the math-rollouts CODE repo
_auth = f"{GH_TOKEN}@" if GH_TOKEN else ""
REPO = f"git+https://{_auth}github.com/{GH_OWNER}/math-rollouts.git"

if is_colab:
    # keep in sync with requirements/colab-gen.txt
    !pip install -q vllm==0.12.0 torch==2.9.0 triton==3.5.0 transformers==4.57.6 tokenizers==0.22.2 huggingface_hub==0.36.2 datasets==4.5.0 pyarrow==23.0.1
    !pip install -q "math_rollouts @ {REPO}"
else:
    !pip install -q "math_rollouts[gen] @ {REPO}"
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Example 1 — K=64 rollouts for MATH-500

<!-- md -->
## Configure

- `MODEL_ID` — any registered model (non-thinking here; for a thinking model see
  Example 3 — it needs a much larger `max_tokens`).
- `POOL` — output pool name → `generations/<slug>/<POOL>.parquet`.
- `K` — rollouts per problem.
- `SEED` — vLLM sampling seed, stored on the rows. An int makes the batch
  reproducible; `None` leaves vLLM unseeded (natural sampling — see Example 3).
  Either way, never REUSE a seed across batches of the same pool: an identical
  request with the same seed regenerates identical completions.

<!-- code -->
```python
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
POOL     = "math500_K64"
K        = 64
SEED     = 0
OUT_ROOT = "/content/math-rollouts-data"
```

<!-- md -->
## Generate + assemble the pool

`generate_natural` loads the problems' prompts via the model's adapter, samples K
completions each on the GPU, and returns raw rollout rows. `pools.build_pool` then
computes the criterion-free facts (`answer_matches`, `has_boxed`, answer placement,
termination, lengths) and conforms the rows to `POOL_SCHEMA`. The tokenizer is
passed through so `answer_token_frac` can be computed.

The headline accuracy is just the default `answer-match` scorer — i.e. the mean of
the `answer_matches` column.

<!-- code -->
```python
from transformers import AutoTokenizer

from math_rollouts.data.problems import load_problems_by_split
from math_rollouts.generate.natural import generate_natural
from math_rollouts.data import pools

problems = load_problems_by_split("math500")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
rows = generate_natural(MODEL_ID, problems, k=K, run_id=0, seed=SEED, tok=tok)

pool_df = pools.build_pool(rows, model_id=MODEL_ID, tok=tok)
acc = pool_df.answer_matches.mean()
print(f"{len(pool_df):,} rollouts over {pool_df.unique_id.nunique()} problems "
      f"| accuracy (answer-match) {acc*100:.1f}%")
```

<!-- md -->
## Write + upload

Writes `<POOL>.parquet` (canonical schema) plus a `<POOL>.meta.json` provenance
sidecar, then uploads both to `generations/<slug>/` in the dataset. The meta records
which named scorer reproduces the headline number (`default_reporting_scorer`).

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
    out.with_suffix(".meta.json"), model_id=MODEL_ID, pool=POOL,
    default_reporting_scorer=pools.default_scorer_id(MODEL_ID),
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
per-prompt sampling params). `ensure_pool_schema` is a no-op on an
already-migrated pool and migrates a legacy one (deriving the answer/match facts)
otherwise — so the extend path works either way.

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

oat_tok = AutoTokenizer.from_pretrained(OAT_MODEL)
existing = pools.ensure_pool_schema(
    load_generation_parquet(OAT_MODEL, OAT_POOL), OAT_MODEL,
    tok=oat_tok, eos_id=oat_tok.eos_token_id)
deficit = pools.pool_deficit(existing, TARGET_K)
print(f"{len(deficit)}/{existing.unique_id.nunique()} problems below K={TARGET_K}; "
      f"generating {sum(deficit.values()):,} new rollouts")

probs = load_problems_by_ids(list(deficit))
run_id = pools.next_run_id(existing)              # fresh batch id (no sample_idx collisions)
new_rows = generate_natural(OAT_MODEL, probs, k=deficit, run_id=run_id, seed=SEED2,
                            tok=oat_tok)
new_df = pools.build_pool(new_rows, model_id=OAT_MODEL, tok=oat_tok)

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
    out.with_suffix(".meta.json"), model_id=OAT_MODEL, pool=OAT_POOL,
    default_reporting_scorer=pools.default_scorer_id(OAT_MODEL),
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
# Example 3 — Thinking model: Qwen3-8B on MATH-500

Qwen3-8B is a **thinking model**: the adapter forces `<think>\n` after the chat
template (so sampling starts at the first *reasoning* token) and the committed
answer is whatever follows `</think>`. Two practical differences from Examples 1–2:

- **Token budget.** Reasoning traces are long — we generate with
  `max_tokens=16384` (vs the 3,000 default). Rollouts that still hit the cap land
  as `terminal == "truncated"` with no committed answer, which is informative, not
  an error; their per-row `max_gen_len` records the budget they had.
- **Scoring.** Nothing extra to do: for thinking adapters the pool's
  `answer_matches` fact is computed on the **post-`</think>` region** (a truncated
  or think-only-boxed rollout is False), so the mean of `answer_matches` *is* the
  `post-think-v1` verdict, and that is what goes in the meta as the
  `default_reporting_scorer`.

This cell is **self-contained** (run Secrets + Install above first): it tops up the
existing `qwen3-8b/math500_natural` pool — migrated from the guided-rollouts
project — to `TARGET_K3` per problem, or bootstraps the pool from scratch if it
isn't on the hub yet.

**Hardware:** Qwen3-8B in bf16 is ~16 GB of weights plus KV cache at an 18k
context — use an A100/H100-class GPU (a T4 won't fit).

<!-- code -->
```python
from math_rollouts.config import GenConfig

QWEN3_MODEL = "Qwen/Qwen3-8B"
QWEN3_POOL  = "math500_natural"   # pool name migrated from guided-rollouts
TARGET_K3   = 64                  # top every problem up to at least this many
# None = UNSEEDED (vLLM samples from fresh entropy) — the natural-sampling
# convention for this pool, and what the legacy rows carry (seed null). Never
# reuse a fixed seed across extension batches of the same pool: re-issuing an
# identical request with the same seed regenerates identical completions.
SEED3       = None
OUT_ROOT    = "/content/math-rollouts-data"

# Qwen3's vendor thinking-mode sampling is T=0.6, top_p=0.95, top_k=20: T/top_p
# match the project defaults, and the adapter's sampling_overrides() adds the
# top_k=20 automatically (the legacy pool's batches were sampled with it — new
# batches must match). So only the budget changes here; max_model_len covers
# prompt + budget.
THINK_CFG = GenConfig(max_tokens=16384, max_model_len=18432)
```

<!-- md -->
## Load the existing pool (or start fresh) and size the deficit

`pool_deficit(..., ids=math500_ids)` counts a problem that is *absent* from the
pool as a full `TARGET_K3` deficit, so partial pools and a missing pool are both
handled by the same path.

<!-- code -->
```python
from transformers import AutoTokenizer
from huggingface_hub.utils import EntryNotFoundError

from math_rollouts.data import pools
from math_rollouts.data.hf import load_generation_parquet
from math_rollouts.data.problems import load_problems_by_ids, load_problems_by_split

math500_ids = [p["unique_id"] for p in load_problems_by_split("math500")]
tok3 = AutoTokenizer.from_pretrained(QWEN3_MODEL)

try:
    existing = pools.ensure_pool_schema(
        load_generation_parquet(QWEN3_MODEL, QWEN3_POOL), QWEN3_MODEL,
        tok=tok3, eos_id=tok3.eos_token_id)
    print(f"existing pool: {len(existing):,} rollouts over "
          f"{existing.unique_id.nunique()} problems")
except EntryNotFoundError:
    existing = None
    print(f"no {QWEN3_POOL} on the hub yet - bootstrapping from scratch")

deficit = (pools.pool_deficit(existing, TARGET_K3, ids=math500_ids)
           if existing is not None else {uid: TARGET_K3 for uid in math500_ids})
run_id = pools.next_run_id(existing)
print(f"{len(deficit)} problems below K={TARGET_K3}; "
      f"generating {sum(deficit.values()):,} rollouts (run_id={run_id})")
```

<!-- md -->
## Generate, assemble, extend

Same flow as Example 2; the only thinking-model knob is `cfg=THINK_CFG`. With
~16k-token traces this is the long pole — budget several GPU-hours for a full
bootstrap (a top-up of a few short problems is much faster).

<!-- code -->
```python
from math_rollouts.generate.natural import generate_natural

probs = load_problems_by_ids(list(deficit))
new_rows = generate_natural(QWEN3_MODEL, probs, k=deficit, run_id=run_id,
                            seed=SEED3, cfg=THINK_CFG, tok=tok3)
new_df = pools.build_pool(new_rows, model_id=QWEN3_MODEL, tok=tok3)
combined = pools.extend_pool(existing, new_df)   # handles existing=None too

acc = combined.answer_matches.mean()             # == post-think-v1 verdict
trunc = (combined.terminal == "truncated").mean()
n_before = 0 if existing is None else len(existing)
print(f"pool: {n_before:,} -> {len(combined):,} rollouts "
      f"| post-think accuracy {acc*100:.1f}% | truncated {trunc*100:.1f}%")
print("min rollouts/problem now:", combined.groupby("unique_id").size().min())
```

<!-- md -->
## Write + upload

The meta's `gen_config` records *this batch's* config; a pool may legitimately mix
budgets across batches (the per-row `temperature`/`top_p`/`max_gen_len` are the
authoritative record — that is what `benchmark@budget` keys off). The
`default_reporting_scorer` resolves to `post-think-v1` for Qwen3.

<!-- code -->
```python
from pathlib import Path
from huggingface_hub import HfApi
from math_rollouts.adapters import get_adapter
from math_rollouts.data.hf import model_slug

slug = model_slug(QWEN3_MODEL)
out = Path(OUT_ROOT) / "generations" / slug / f"{QWEN3_POOL}.parquet"
pools.write_pool(combined, out)
pools.write_pool_meta(
    out.with_suffix(".meta.json"), model_id=QWEN3_MODEL, pool=QWEN3_POOL,
    default_reporting_scorer=pools.default_scorer_id(QWEN3_MODEL),  # post-think-v1
    gen_config=dict(THINK_CFG.as_dict(),
                    sampling_overrides=get_adapter(QWEN3_MODEL).sampling_overrides()),
    runs=[{"run_id": int(r), "n_rollouts": int((combined.run_id == r).sum())}
          for r in sorted(combined.run_id.unique())],
    df=combined,
)

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/math-rollouts"
for f in (out, out.with_suffix(".meta.json")):
    api.upload_file(path_or_fileobj=str(f), repo_id=repo_id, repo_type="dataset",
                    path_in_repo=f"generations/{slug}/{f.name}",
                    commit_message=f"Extend {slug}/{QWEN3_POOL} to >=K={TARGET_K3} "
                                   f"({len(combined)} rollouts)")
print("uploaded", f"generations/{slug}/{QWEN3_POOL}.parquet")
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
## Notes — truncation & length-extend (forward-looking)

- **Truncation / budget.** `terminal` (`emitted_eos` vs `truncated`) and the per-row
  `max_gen_len` record how each rollout ended. Because they're per-row, a pool can
  mix budgets — which sets up a future **length-extend**: re-generate the
  `terminal == "truncated"` rows at a higher cap and update them in place by
  `ROLLOUT_KEY` (`model_id, unique_id, run_id, branch_path, sample_idx`), bumping
  `max_gen_len`/`finish_reason`. (This notebook implements the *width*-extend —
  more rollouts per problem.)
- **Other thinking models** (e.g. DeepSeek-R1-Distill-Qwen-1.5B) work through the
  same Example-3 flow — just swap the model id; the adapter registry picks the right
  prompt template and `</think>` handling.
