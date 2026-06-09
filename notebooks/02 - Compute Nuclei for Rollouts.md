<!-- code -->
```python
# Defer annotation evaluation so helpers can be defined in their own cells
# above the Setup cell (which is where ``Path`` etc. are actually imported).
# In notebook execution this doesn't matter -- cells run top-to-bottom -- but
# running the file as a plain script otherwise hits a def-time NameError on
# the ``path: Path`` annotation of ``_write_html_file``.
from __future__ import annotations
```

<!-- md -->
# Inside the Sampling Nucleus — How Small Is It?

<!-- md -->

For each generated token, nucleus (top-p) sampling keeps only the smallest set of
tokens whose probability mass reaches `p` — the **nucleus** — and the model can
only ever sample from that set. A striking property of these math rollouts is how
*small* the nucleus usually is: at most positions it collapses to a **single
token** (the model is effectively deterministic there).

This notebook quantifies that over a whole pool of naturally-sampled rollouts from
the [`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)
dataset. For every rollout it teacher-forces the completion back through the model
and records, at each generated position, the **nucleus size** (and whether the
token the rollout took was the model's top-1). It then reports the singleton
fraction and the full size distribution, and pushes the per-token results back to
the dataset.

The heavy lifting lives in the `math-rollouts` package
(`math_rollouts.analysis.token_nuclei`); this notebook just wires up secrets,
installs the code, runs it on a GPU, and uploads the results.

<!-- md -->
Check for Colab vs. script.

This file can also be run as a script, so we need to guard some actions to only happen when we're running from within a Colab Notebook.

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
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Secrets

Pulled from Colab's **userdata** (the key icon in the left sidebar). You need:

- `HF_TOKEN` — a Hugging Face token with **write** access (to push results).
- `HF_USERNAME` — your HF username; the dataset is `<HF_USERNAME>/math-rollouts`.
- `GITHUB_TOKEN` — only needed if the `math-rollouts` code repo is private.

Outside Colab we fall back to environment variables.

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

# huggingface_hub picks this up automatically for the dataset pull.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

assert HF_USERNAME, "set HF_USERNAME (Colab userdata or env var)"
print("HF user:", HF_USERNAME, "| HF token:", "set" if HF_TOKEN else "MISSING")
```

<!-- output -->
```
HF user: ChrisMcCormick | HF token: set
```

<!-- md -->
# Install the `math-rollouts` package

Installs the code plus its light CPU dependencies (numpy / pandas / pyarrow /
anytree / huggingface_hub / datasets). `torch` and `transformers` are already
present in Colab, so we deliberately do **not** install the `[gen]` extra (which
would pull vLLM). The GitHub token, if present, is injected into the URL so this
works for a private repo too; it's harmless for a public one.

<!-- code -->
```python
GH_OWNER = "chrisjmccormick"   # GitHub owner of the math-rollouts CODE repo
_auth = f"{GH_TOKEN}@" if GH_TOKEN else ""
pip_url = f"git+https://{_auth}github.com/{GH_OWNER}/math-rollouts.git"
!pip install -q {pip_url}
```

<!-- output -->
```
  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
  Preparing metadata (pyproject.toml) ... [?25l[?25hdone
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Qwen2.5-Math-1.5B

<!-- md -->
## Configure

- `POOL` — which naturally-sampled pool to analyze. 
    - `math12k_L4_5_K64` (64
  rollouts per problem, uniform across all difficulty levels)
    - `math500_passK` (~40k
  rollouts over the 500 MATH-500 problems)
    - `math12k_passK` (~130k) gives a tighter estimate but 
- `LIMIT` — cap the number of rollouts. `None` processes the whole pool; a few
  thousand already pins the singleton fraction tightly, so set e.g. `2000` for a
  quick pass first.
- `SHARD_SIZE` — problems per output parquet. `1` (per-problem) is right for
  `math500_passK` (500 small files the viz can load one at a time); bump it (e.g.
  `50`) for the larger `math12k` pools to avoid thousands of tiny files.
- `LOGIT_DTYPE` — `float16` (default, halves the logit bytes and matches the bf16
  compute precision) or `float32` if your `pyarrow` can't write float16.

<!-- code -->
```python
MODEL_ID    = "Qwen/Qwen2.5-Math-1.5B"
POOL        = "math500_passK"
LIMIT       = None
SHARD_SIZE  = 1
LOGIT_DTYPE = "float16"
TOP_K       = None   # nucleus-size storage cap (None = GenConfig default 20). NOT a sampling limiter — the rollouts had none; raise to record larger nuclei
OUT_ROOT    = "/content/math-rollouts-data"
```

<!-- md -->
## Compute the per-token nuclei

<!-- md -->

`build_token_nuclei` pulls the pool + problem text from the HF dataset (cached
locally on first use), teacher-forces every rollout on the GPU in length-packed
batches, and at each generated token records the nucleus size plus a frugal slice
of the distribution: **top-2** for singletons (so you can see if the runner-up had
any mass) and **10–20** for branch tokens (the nucleus plus a few alternates just
outside it). Per kept entry it stores the raw logit + token id.

It writes per-problem shards under
`OUT_ROOT/generations/<model-slug>/<POOL>_token_nuclei/`, plus `_stats.json` (the
headline numbers) and `_meta.json` (model / engine / dtype / keep-rule), and
returns `(stats, paths)`.

Recompute precision is **bfloat16**, matching the vLLM engine that generated the
rollouts (some first-token logits are nearly tied, so fp32 would reshuffle the
nucleus).

<!-- code -->
```python
from math_rollouts.analysis.token_nuclei import build_token_nuclei

stats, paths = build_token_nuclei(
    MODEL_ID, POOL, OUT_ROOT,
    limit=LIMIT,
    shard_size=SHARD_SIZE,
    logit_dtype=LOGIT_DTYPE,
    top_k=TOP_K,
    device="cuda",
)
```

<!-- output -->
```
Loading Qwen/Qwen2.5-Math-1.5B on cuda (bfloat16) ...
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]
500 problems, 40704 rollouts -> shards of 1 problem(s)
  50/500 problems, 3456/40704 rollouts
  100/500 problems, 7168/40704 rollouts
  150/500 problems, 11136/40704 rollouts
  200/500 problems, 15616/40704 rollouts
  250/500 problems, 21120/40704 rollouts
  300/500 problems, 26368/40704 rollouts
  350/500 problems, 29568/40704 rollouts
  400/500 problems, 33024/40704 rollouts
  450/500 problems, 36736/40704 rollouts
  500/500 problems, 40704/40704 rollouts

=== nucleus store: Qwen/Qwen2.5-Math-1.5B / math500_passK ===
  rollouts: 40,704   tokens: 61,831,319
  SINGLETON nuclei: 93.3% (57,693,565 / 61,831,319)
  full size / difficulty / correctness stats: compute post-hoc from the shards via
  math_rollouts.analysis.nuclei_stats.summarize_nuclei (see notebook '03 - Analyze Rollout Nuclei').

wrote 500 shards + _stats.json + _meta.json to /content/math-rollouts-data/generations/qwen2.5-math-1.5b/math500_passK_token_nuclei
```

<!-- md -->
## Inspect the statistics

<!-- md -->

`build_token_nuclei` now returns only the **headline counts** (rollouts, tokens,
singleton %). The full size distribution, per-difficulty bands, position profile, and
correct/incorrect splits are computed post-hoc from the uploaded shards in the
companion notebook **`03 - Analyze Rollout Nuclei`** (via
`analysis.nuclei_stats.summarize_nuclei`) — keeping this compute notebook lean.

<!-- code -->
```python
import json
print(json.dumps(stats, indent=2))
```

<!-- output -->
```
{
  "n_rollouts": 40704,
  "n_tokens": 61831319,
  "singleton_count": 57693565,
  "singleton_frac": 0.9330799655106824
}
```

<!-- md -->
## Push results to the HF dataset

<!-- md -->

Uploads the whole `<POOL>_token_nuclei/` shard directory (per-problem parquets +
`_stats.json` + `_meta.json`) to `generations/<model-slug>/` in
`<HF_USERNAME>/math-rollouts`. Needs the write-scoped `HF_TOKEN` from the Secrets
cell.

<!-- code -->
```python
from huggingface_hub import HfApi
from math_rollouts.data.hf import model_slug

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/math-rollouts"
slug = model_slug(MODEL_ID)
dest = f"generations/{slug}/{paths['dir'].name}"

api.upload_folder(
    folder_path=str(paths["dir"]),
    path_in_repo=dest,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Add per-token nucleus store for {POOL} ({stats['n_rollouts']} rollouts)",
)
print("uploaded", dest)
```

<!-- output -->
```
It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
WARNING:huggingface_hub.hf_api:It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
Processing Files (0 / 0)      : |          |  0.00B /  0.00B
New Data Upload               : |          |  0.00B /  0.00B
  ...h500_algebra_7506.parquet:   1%|          | 3.66kB /  465kB
  ..._probability_8758.parquet:   1%|          | 2.89kB /  367kB
  ...h500_algebra_8101.parquet:   1%|          | 3.01kB /  383kB
  ...h500_algebra_8115.parquet:   1%|          | 2.09kB /  265kB
  ...h500_algebra_8588.parquet:   1%|          | 3.62kB /  459kB
  ...h500_algebra_7636.parquet:   1%|          | 3.02kB /  383kB
  ...h500_algebra_8353.parquet:   1%|          | 3.42kB /  434kB
  ...h500_algebra_8122.parquet:   1%|          | 4.43kB /  562kB
  ...h500_algebra_7564.parquet:   1%|          | 3.47kB /  440kB
  ...h500_algebra_8359.parquet:   1%|          | 3.55kB /  451kB
uploaded generations/qwen2.5-math-1.5b/math500_passK_token_nuclei
```

<!-- md -->
TODO:

> WARNING:huggingface_hub.hf_api:It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
P

<!-- md -->
Singleton fraction **by difficulty band** (and the rest of the breakdowns) now lives
in `03 - Analyze Rollout Nuclei`, computed from these uploaded shards with an even
K-per-problem sample so the pass@K difficulty skew doesn't confound the numbers.

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Oat-Zero

<!-- md -->
## Configure

<!-- md -->

- `POOL` — which naturally-sampled pool to analyze. `math500_passK` (\~40k
  rollouts over the 500 MATH-500 problems) is a good default; `math12k_passK`
  (~130k) gives a tighter estimate but takes longer.
- `LIMIT` — cap the number of rollouts. `None` processes the whole pool; a few
  thousand already pins the singleton fraction tightly, so set e.g. `2000` for a
  quick pass first.
- `SHARD_SIZE` — problems per output parquet. `1` (per-problem) is right for
  `math500_passK` (500 small files the viz can load one at a time); bump it (e.g.
  `50`) for the larger `math12k` pools to avoid thousands of tiny files.
- `LOGIT_DTYPE` — `float16` (default, halves the logit bytes and matches the bf16
  compute precision) or `float32` if your `pyarrow` can't write float16.

<!-- code -->
```python
# https://huggingface.co/datasets/ChrisMcCormick/math-rollouts/blob/main/generations/qwen2.5-math-1.5b-oat-zero/math500_passK.parquet
MODEL_ID    = "sail/Qwen2.5-Math-1.5B-Oat-Zero"
POOL        = "math500_passK"
LIMIT       = None
SHARD_SIZE  = 1
LOGIT_DTYPE = "float16"
OUT_ROOT    = "/content/math-rollouts-data"
```

<!-- md -->
## Compute the per-token nuclei

<!-- md -->

`build_token_nuclei` pulls the pool + problem text from the HF dataset (cached
locally on first use), teacher-forces every rollout on the GPU in length-packed
batches, and at each generated token records the nucleus size plus a frugal slice
of the distribution: **top-2** for singletons (so you can see if the runner-up had
any mass) and **10–20** for branch tokens (the nucleus plus a few alternates just
outside it). Per kept entry it stores the raw logit + token id.

It writes per-problem shards under
`OUT_ROOT/generations/<model-slug>/<POOL>_token_nuclei/`, plus `_stats.json` (the
headline numbers) and `_meta.json` (model / engine / dtype / keep-rule), and
returns `(stats, paths)`.

Recompute precision is **bfloat16**, matching the vLLM engine that generated the
rollouts (some first-token logits are nearly tied, so fp32 would reshuffle the
nucleus).

<!-- code -->
```python
from math_rollouts.analysis.token_nuclei import build_token_nuclei

stats, paths = build_token_nuclei(
    MODEL_ID, POOL, OUT_ROOT,
    limit=LIMIT,
    shard_size=SHARD_SIZE,
    logit_dtype=LOGIT_DTYPE,
    device="cuda",
)
```

<!-- output -->
```
config.json:   0%|          | 0.00/859 [00:00<?, ?B/s]
tokenizer_config.json:   0%|          | 0.00/7.35k [00:00<?, ?B/s]
vocab.json:   0%|          | 0.00/2.78M [00:00<?, ?B/s]
merges.txt:   0%|          | 0.00/1.67M [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/11.4M [00:00<?, ?B/s]
added_tokens.json:   0%|          | 0.00/605 [00:00<?, ?B/s]
special_tokens_map.json:   0%|          | 0.00/616 [00:00<?, ?B/s]
Loading sail/Qwen2.5-Math-1.5B-Oat-Zero on cuda (bfloat16) ...
model.safetensors:   0%|          | 0.00/3.09G [00:00<?, ?B/s]
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]
generation_config.json:   0%|          | 0.00/117 [00:00<?, ?B/s]
generations/qwen2.5-math-1.5b-oat-zero/m(…):   0%|          | 0.00/17.0M [00:00<?, ?B/s]
500 problems, 21312 rollouts -> shards of 1 problem(s)
  50/500 problems, 1312/21312 rollouts
  100/500 problems, 2624/21312 rollouts
  150/500 problems, 4704/21312 rollouts
  200/500 problems, 7552/21312 rollouts
  250/500 problems, 11168/21312 rollouts
  300/500 problems, 14784/21312 rollouts
  350/500 problems, 15840/21312 rollouts
  400/500 problems, 17152/21312 rollouts
  450/500 problems, 19232/21312 rollouts
  500/500 problems, 21312/21312 rollouts

=== nucleus store: sail/Qwen2.5-Math-1.5B-Oat-Zero / math500_passK ===
  rollouts: 21,312   tokens: 18,387,973
  SINGLETON nuclei: 94.6% (17,391,801 / 18,387,973)
  full size / difficulty / correctness stats: compute post-hoc from the shards via
  math_rollouts.analysis.nuclei_stats.summarize_nuclei (see notebook '03 - Analyze Rollout Nuclei').

wrote 500 shards + _stats.json + _meta.json to /content/math-rollouts-data/generations/qwen2.5-math-1.5b-oat-zero/math500_passK_token_nuclei
```

<!-- md -->
## Inspect the statistics

<!-- md -->

Headline counts only — the full size distribution and breakdowns are computed
post-hoc in `03 - Analyze Rollout Nuclei` from the uploaded shards.

<!-- code -->
```python
import json
print(json.dumps(stats, indent=2))
```

<!-- output -->
```
{
  "n_rollouts": 21312,
  "n_tokens": 18387973,
  "singleton_count": 17391801,
  "singleton_frac": 0.9458248062469964
}
```

<!-- md -->
## Push results to the HF dataset

<!-- md -->

Uploads the whole `<POOL>_token_nuclei/` shard directory (per-problem parquets +
`_stats.json` + `_meta.json`) to `generations/<model-slug>/` in
`<HF_USERNAME>/math-rollouts`. Needs the write-scoped `HF_TOKEN` from the Secrets
cell.

<!-- code -->
```python
from huggingface_hub import HfApi
from math_rollouts.data.hf import model_slug

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_USERNAME}/math-rollouts"
slug = model_slug(MODEL_ID)
dest = f"generations/{slug}/{paths['dir'].name}"

api.upload_folder(
    folder_path=str(paths["dir"]),
    path_in_repo=dest,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Add per-token nucleus store for {POOL} ({stats['n_rollouts']} rollouts)",
)
print("uploaded", dest)
```

<!-- output -->
```
It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
WARNING:huggingface_hub.hf_api:It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
Processing Files (0 / 0)      : |          |  0.00B /  0.00B
New Data Upload               : |          |  0.00B /  0.00B
  ..._prealgebra_11765.parquet:   1%|          |   464B / 57.9kB
  ...h500_algebra_8546.parquet:   1%|          |   388B / 48.4kB
  ..._prealgebra_11655.parquet:   1%|          |   124B / 15.5kB
  ...h500_algebra_7770.parquet:   1%|          |   432B / 54.0kB
  ...h500_algebra_8441.parquet:   1%|          |   179B / 22.4kB
  ..._prealgebra_11722.parquet:   1%|          |   202B / 25.2kB
  ...500_geometry_9380.parquet:   1%|          |   474B / 59.2kB
  ...iate_algebra_9738.parquet:   1%|          |   330B / 41.2kB
  ...h500_algebra_8340.parquet:   1%|          |   213B / 26.7kB
  ...h500_algebra_8628.parquet:   1%|          |   186B / 23.3kB
uploaded generations/qwen2.5-math-1.5b-oat-zero/math500_passK_token_nuclei
```

<!-- md -->
TODO:

> WARNING:huggingface_hub.hf_api:It seems you are trying to upload a large folder at once. This might take some time and then fail if the folder is too large. For such cases, it is recommended to upload in smaller batches or to use `HfApi().upload_large_folder(...)`/`hf upload-large-folder` instead. For more details, check out https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#upload-a-large-folder.
P

