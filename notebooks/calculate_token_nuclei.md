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

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Configure

- `POOL` — which naturally-sampled pool to analyze. `math12k_L4_5_K64` (64
  rollouts per problem, uniform across all difficulty levels) is the best choice
  for difficulty-band analysis; `math500_passK` (~40k rollouts, but skewed toward
  hard problems) or `math12k_passK` (~130k) are alternatives.
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
POOL        = "math12k_L4_5_K64"
LIMIT       = None
SHARD_SIZE  = 1
LOGIT_DTYPE = "float16"
TOP_K       = None   # None = use GenConfig default (20, matching original sampling); raise to see larger nuclei
OUT_ROOT    = "/content/math-rollouts-data"
```

<!-- md -->
# Compute the per-token nuclei

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

<!-- md -->
# Inspect the statistics

The full summary as JSON, plus a histogram of nucleus sizes across all generated
tokens. The first bar (size 1) is the singleton fraction — the headline number for
the blog post.

<!-- code -->
```python
import json
print(json.dumps(stats, indent=2))
```

<!-- code -->
```python
import matplotlib.pyplot as plt

hist = stats["size_histogram"]
sizes = sorted(hist)
pct = [hist[s] / stats["n_tokens"] * 100 for s in sizes]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(sizes, pct, color="#4C72B0")
plt.bar_label(bars, labels=[f"{p:.0f}%" if p >= 1 else "" for p in pct], fontsize=8)
plt.xlabel("nucleus size (number of tokens sampling could pick)")
plt.ylabel("% of generated tokens")
plt.title(f"Nucleus size distribution — {MODEL_ID.split('/')[-1]} / {POOL}\n"
          f"{stats['singleton_frac']*100:.1f}% of tokens have a singleton nucleus")
plt.xticks(sizes)
plt.tight_layout()
plt.show()
```

<!-- code -->
```python
# Singleton fraction by difficulty band (requires difficulty data to be registered
# for this model; the by_band dict is empty otherwise).
by_band = stats.get("by_band", {})
if by_band:
    bands  = list(by_band.keys())
    fracs  = [by_band[b]["singleton_frac"] * 100 for b in bands]
    tokens = [by_band[b]["n_tokens"] for b in bands]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#8e44ad"]
    bars = ax.bar(bands, fracs, color=colors[:len(bands)])
    ax.bar_label(bars, labels=[f"{f:.1f}%" for f in fracs], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("singleton nucleus % of tokens")
    ax.set_title(f"Singleton fraction by difficulty — {MODEL_ID.split('/')[-1]} / {POOL}")
    for bar, n in zip(bars, tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, 2, f"{n:,}\ntokens",
                ha="center", va="bottom", fontsize=7, color="white")
    plt.tight_layout()
    plt.show()
else:
    print("No by_band data — difficulty not registered for this model.")
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Push results to the HF dataset

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
