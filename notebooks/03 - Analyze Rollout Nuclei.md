<!-- code -->
```python
from __future__ import annotations
```

<!-- md -->
# Analyzing the Sampling Nucleus — Size, Difficulty, and Position

This is the **analysis** companion to `02 - Compute Nuclei for Rollouts`. That
notebook teacher-forces every rollout through the model on a GPU and writes a
per-token **nucleus store** to the
[`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)
dataset. Here we just *pull that store* (no GPU, no model) and reshape it to answer
report questions:

- Overall nucleus-size statistics and the **singleton fraction**.
- How those vary **by problem difficulty**.
- **Branch tokens per 1k** — the rate at which sampling can actually diverge.
- How nucleus size depends on **sequence position**.
- A **correct vs. incorrect** comparison (with the confounds controlled).

**De-confounding difficulty.** `math500_passK` is a *pass@K* corpus: harder problems
carry more rollouts (up to 320) than easy ones (64). A raw token histogram would be
weighted toward hard problems. We fix this by taking an **even K=64 rollouts per
problem**, so every MATH-500 problem contributes equally.

The aggregation lives in the `math_rollouts` package
(`analysis.nuclei_stats`, `data.hf.load_token_nuclei_pool`,
`analysis.difficulty`); this notebook is just the report-specific wiring.

<!-- md -->
Check for Colab vs. script — guard Colab-only actions so this file can also run as a
plain script.

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
# Install the `math-rollouts` package

Analysis-only, so we install the **CPU** dependencies (numpy / pandas / pyarrow /
huggingface_hub / datasets) and deliberately **skip** the `[gen]` extra (torch /
transformers / vLLM) — nothing here loads a model. The GitHub token, if present, is
injected for a private code repo; it's harmless for a public one.

<!-- code -->
```python
import os

GH_OWNER = "chrisjmccormick"   # GitHub owner of the math-rollouts CODE repo
GH_TOKEN = None
if is_colab:
    try:
        GH_TOKEN = userdata.get("GITHUB_TOKEN")
    except Exception:
        GH_TOKEN = None
    # A read-scoped HF token (optional) just raises the hub rate limit; the dataset
    # is public, so reads work without it.
    try:
        os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
    except Exception:
        pass
    _auth = f"{GH_TOKEN}@" if GH_TOKEN else ""
    !pip install -q "git+https://{_auth}github.com/{GH_OWNER}/math-rollouts.git"
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Configure

- `MODEL_ID` — which model's nucleus store to analyze.
- `POOL` — naturally-sampled pool. `math500_passK` is the pass@K MATH-500 corpus.
- `K_PER_PROBLEM` — even rollouts per problem. **64** keeps all 500 MATH-500
  problems (every one has ≥64); lower it (e.g. 16) only if a pool has thinner
  problems.
- `CI_BAND` — difficulty band used for the correct-vs-incorrect comparison (one with
  a healthy mix of both outcomes).

<!-- code -->
```python
MODEL_ID      = "Qwen/Qwen2.5-Math-1.5B"
POOL          = "math500_passK"
K_PER_PROBLEM = 64
SEED          = 0
CI_BAND       = "Hard"

# Only the light columns — skip the bulky kept_ids/kept_logits lists, we only need sizes.
LIGHT_COLS = ["unique_id", "subject", "sample_idx", "is_correct", "n_tokens",
              "nuc_sizes", "chosen_is_top1"]
```

<!-- md -->
# Load the nucleus store

`load_token_nuclei_pool` pulls every per-problem shard for the pool from the dataset
(`snapshot_download`, cached locally on first use) and concatenates them into one
DataFrame — **one row per rollout**, each carrying the per-token `nuc_sizes` list.

<!-- code -->
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math_rollouts.data.hf import load_token_nuclei_pool
from math_rollouts.analysis.difficulty import band_table, BAND_ORDER
from math_rollouts.analysis.nuclei_stats import (
    summarize_nuclei, even_k_sample, size_by_position)

df = load_token_nuclei_pool(MODEL_ID, POOL, columns=LIGHT_COLS)
print(f"loaded {len(df):,} rollouts across {df.unique_id.nunique()} problems "
      f"({POOL} / {MODEL_ID})")
```

<!-- md -->
# Even-K sampling

The pass@K skew, made concrete: rollouts-per-problem and how many problems survive
each K. We then subsample to an even **K per problem** so the size statistics weight
every problem equally instead of every rollout.

<!-- code -->
```python
kpp = df.groupby("unique_id").size()
print(f"rollouts per problem: min {kpp.min()}  median {int(kpp.median())}  max {kpp.max()}")
for thr in (16, 32, 64):
    print(f"  problems with >= {thr:>2} rollouts: {int((kpp >= thr).sum())}/{len(kpp)}")

bal = even_k_sample(df, K_PER_PROBLEM, seed=SEED)
kept = bal.unique_id.nunique()
print(f"\neven-K={K_PER_PROBLEM}: kept {kept}/{len(kpp)} problems, {len(bal):,} rollouts, "
      f"{int(bal.n_tokens.sum()):,} tokens "
      f"(dropped {len(kpp) - kept} with < {K_PER_PROBLEM} rollouts)")
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Overall nucleus-size statistics

`summarize_nuclei` recomputes the headline numbers post-hoc from the stored sizes,
so we can run it on any slice. We compare the **balanced** sample to the raw pool to
see how much the pass@K difficulty skew was inflating things.

<!-- code -->
```python
stats = summarize_nuclei(bal)
raw   = summarize_nuclei(df)
br1k  = lambda s: (1 - s["singleton_frac"]) * 1000

print(f"=== balanced (even-K={K_PER_PROBLEM}) ===")
print(f"  rollouts {stats['n_rollouts']:,}   tokens {stats['n_tokens']:,}")
print(f"  SINGLETON: {stats['singleton_frac']*100:.1f}%   "
      f"branch/1k: {br1k(stats):.1f}   ({br1k(stats)/10:.1f}% of tokens are branch points)")
print(f"  mean {stats['mean_size']:.3f}  median {stats['median_size']}  p90 {stats['p90_size']}")
print(f"  chose top-1: {stats['chosen_is_top1_frac']*100:.1f}%")
print(f"\n  (raw unbalanced pool singleton: {raw['singleton_frac']*100:.1f}% over "
      f"{raw['n_tokens']:,} tokens — pass@K-weighted toward hard problems)")
```

<!-- md -->
The size distribution across all generated tokens. The first bar (size 1) is the
singleton fraction — the headline number for the post.

<!-- code -->
```python
hist  = stats["size_histogram"]
sizes = sorted(hist)
pct   = [hist[s] / stats["n_tokens"] * 100 for s in sizes]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(sizes, pct, color="#4C72B0")
plt.bar_label(bars, labels=[f"{p:.0f}%" if p >= 1 else "" for p in pct], fontsize=8)
plt.xlabel("nucleus size (number of tokens sampling could pick)")
plt.ylabel("% of generated tokens")
plt.title(f"Nucleus size distribution — {MODEL_ID.split('/')[-1]} / {POOL} (even-K={K_PER_PROBLEM})\n"
          f"{stats['singleton_frac']*100:.1f}% of tokens have a singleton nucleus")
plt.xticks(sizes)
plt.tight_layout()
plt.show()
```

<!-- md -->
# Per-difficulty statistics

Difficulty is the model's **empirical solve rate** per problem, banded by
`analysis.difficulty` (`Easy`…`Impossible`). Passing a `band_map` to
`summarize_nuclei` gives a per-band breakdown over the same balanced sample.

<!-- code -->
```python
bands    = band_table(MODEL_ID)                       # unique_id, acc, n, band
band_map = dict(zip(bands.unique_id, bands.band))
by_band  = summarize_nuclei(bal, band_map=band_map)["by_band"]

n_prob = bal.assign(band=bal.unique_id.map(band_map)).groupby("band").unique_id.nunique()
order  = [b for b in BAND_ORDER if b in by_band] + [b for b in by_band if b not in BAND_ORDER]
band_df = pd.DataFrame([{
    "band": b,
    "n_problems": int(n_prob.get(b, 0)),
    "n_tokens": by_band[b]["n_tokens"],
    "singleton_%": round(by_band[b]["singleton_frac"] * 100, 1),
    "branch_/1k": round((1 - by_band[b]["singleton_frac"]) * 1000, 1),
    "mean_size": round(by_band[b]["mean_size"], 3),
    "top1_%": round(by_band[b]["chosen_is_top1_frac"] * 100, 1),
} for b in order])
print(band_df.to_string(index=False))
```

<!-- md -->
Branch rate by difficulty. It's **U-shaped**, not monotonic: the model branches most
on the two extremes — easy problems it cruises through and impossible ones it has no
foothold on — and is *most* deterministic on the hard-but-attempted middle.

<!-- code -->
```python
colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#8e44ad"]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(band_df["band"], band_df["branch_/1k"], color=colors[:len(band_df)])
ax.bar_label(bars, labels=[f"{v:.0f}" for v in band_df["branch_/1k"]], fontsize=9)
ax.set_ylabel("branch tokens per 1k")
ax.set_title(f"Branch rate by difficulty — {MODEL_ID.split('/')[-1]} (even-K={K_PER_PROBLEM})")
for bar, n in zip(bars, band_df["n_problems"]):
    ax.text(bar.get_x() + bar.get_width() / 2, 2, f"{n}\nprob", ha="center",
            va="bottom", fontsize=7, color="white")
plt.tight_layout()
plt.show()
```

<!-- md -->
# ▂▂▂▂▂▂▂▂▂▂▂▂

<!-- md -->
# Nucleus size vs. sequence position

Where in the generation does the branching actually happen? `size_by_position`
averages nucleus size (and singleton fraction) across rollouts at each token
position. `count[i]` falls off as fewer rollouts reach deep positions, so the tail is
noisier and averaged over the longer rollouts.

<!-- code -->
```python
mean_pos, sing_pos, cnt_pos = size_by_position(bal)

buckets = [(0, 1), (1, 2), (2, 10), (10, 50), (50, 200), (200, 800), (800, len(cnt_pos))]
pos_rows = []
for lo, hi in buckets:
    c = cnt_pos[lo:hi]
    n = int(c.sum())
    if not n:
        continue
    label = f"{lo}" if hi - lo == 1 else f"{lo}-{min(hi, len(cnt_pos)) - 1}"
    pos_rows.append({
        "position": label, "n_tokens": n,
        "mean_size": round(float((mean_pos[lo:hi] * c).sum() / n), 3),
        "singleton_%": round(float((sing_pos[lo:hi] * c).sum() / n) * 100, 1),
    })
print(pd.DataFrame(pos_rows).to_string(index=False))
```

<!-- md -->
The curve over the first ~512 positions. Branching is heavily **front-loaded**: the
opening token is a genuine fork (mean nucleus ≈ 6.5), then the model collapses toward
near-determinism within a few hundred tokens.

<!-- code -->
```python
P = 512
x = np.arange(P)
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(x, mean_pos[:P], color="#4C72B0", label="mean nucleus size")
ax1.set_xlabel("token position in completion")
ax1.set_ylabel("mean nucleus size", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")

ax2 = ax1.twinx()
ax2.plot(x, sing_pos[:P] * 100, color="#e67e22", label="singleton %")
ax2.set_ylabel("singleton %", color="#e67e22")
ax2.tick_params(axis="y", labelcolor="#e67e22")
ax2.set_ylim(0, 100)

plt.title(f"Nucleus size vs. position — {MODEL_ID.split('/')[-1]} (even-K={K_PER_PROBLEM}, first {P} tokens)")
plt.tight_layout()
plt.show()
```

<!-- md -->
# Correct vs. incorrect rollouts

Naively, incorrect rollouts look *more* deterministic — but that's confounded:
incorrect rollouts are ~2.5× longer (and late tokens are near-singleton, per the
position curve above) and skew to hard problems. We control for difficulty by
splitting **within a single band**; a residual length gap remains, so read it as
"correct rollouts branch somewhat more, even at fixed difficulty."

<!-- code -->
```python
def ci_table(frame, title):
    rows = []
    for label in ("correct", "incorrect"):
        sub = frame[frame.is_correct] if label == "correct" else frame[~frame.is_correct]
        if not len(sub):
            continue
        s = summarize_nuclei(sub)
        rows.append({
            "rollouts": label, "n": s["n_rollouts"], "n_tokens": s["n_tokens"],
            "singleton_%": round(s["singleton_frac"] * 100, 1),
            "branch_/1k": round(br1k(s), 1),
            "mean_size": round(s["mean_size"], 3),
            "top1_%": round(s["chosen_is_top1_frac"] * 100, 1),
        })
    print(f"=== {title} ===")
    print(pd.DataFrame(rows).to_string(index=False), "\n")

bal_band = bal.assign(band=bal.unique_id.map(band_map))
ci_table(bal, f"pooled, even-K={K_PER_PROBLEM} (confounded by length + difficulty)")
ci_table(bal_band[bal_band.band == CI_BAND], f"within '{CI_BAND}' band (difficulty held fixed)")
```
