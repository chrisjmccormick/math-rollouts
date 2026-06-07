# math-rollouts

Naturally-sampled math reasoning rollouts for **Qwen2.5-Math-1.5B** (and, by
design, future thinking/non-thinking models), plus first-token / branch **nuclei**
and uniform-opener rollouts — with a single unified generator and a separate,
re-runnable scoring pass.

**Code lives here (GitHub); data lives in the HF dataset
[`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)
(`repo_type=dataset`).** This split keeps git light, lets data consumers (e.g. the
separate `nucleus-viz` repo) pull parquets independently, and keeps generation
reproducible from code.

This artifact is **guidance-free**: only naturally-sampled rollouts from public
checkpoints. Excluded are models *we* fine-tuned in-house and teacher-guided
(intersection-sampling) rollouts. Significant public models — base
Qwen2.5-Math-1.5B, Oat-Zero, DeepSeek-R1-Distill, Qwen3 — are all in scope.

## What's where

| | |
|---|---|
| **Code (this repo)** | unified generator, model adapters, nucleus tree, scorers, analysis (accuracy tables, difficulty banding), id mapping |
| **Data (HF dataset)** | `problems/`, `mappings/`, `generations/<model-slug>/<experiment>/` parquets + manifests |

```
generations/<model-slug>/<experiment>/
  nuclei.parquet     one row per opener (leaf of the nucleus tree)
  rollouts.parquet   RAW forced samples — NO correctness column
  scores.parquet     DERIVED, versioned by scorer_id — re-runnable on CPU
  policies.csv        opener-policy accuracy summary
  manifest.json       model_id, GenConfig, git sha, counts
```

## Core ideas

- **One code path for nuclei.** A first-token nucleus is just a depth-1 branch
  tree. `NucleusTree(max_depth=1)` reproduces the legacy `openings_k16` first-token
  recipe byte-for-byte; `max_depth>1` walks deeper branches with one persistent KV
  cache (DFS, `cache.crop()` on backtrack). See `nucleus/tree.py`.
- **Model differences live in one ABC.** `adapters/base.py:ModelAdapter` captures
  the only thinking/non-thinking differences (prompt up to the nucleus root,
  terminal tokens, family-default scoring, vLLM stop strings). Adding a model =
  one subclass + one line in `adapters/registry.py`. No schema or generator change.
- **Generation and scoring are separated.** Generation writes pristine RAW rollouts
  (GPU). Scoring (`score/run.py`, CPU) reads them and writes `scores.parquet` under
  a versioned `scorer_id`, so accuracy can be recomputed under a different scorer
  (post-`</think>`, leak-filter `KEEP_FRAC`, …) without re-running generation.
- **Grouping is explicit.** The "these K were generated together" key is
  `(model_id, unique_id, branch_path, run_id)`; `accuracy = Σ is_correct /
  group_size` where `group_size` is the row count — never a stored, fragile count.
  `branch_path` (child-index at each fork) is the durable opener identity.

## Install

```bash
pip install -e .            # CPU: scoring, analysis, data loading
pip install -e '.[gen]'     # + torch/transformers/vllm for GPU generation
pip install -e '.[dev]'     # + pytest
```

## Regenerate the MATH-500 uniform openers (the first job)

Depth-1 nucleus, base Qwen2.5-Math-1.5B, K=16 forced uniform rollouts, all 500:

```bash
# GPU box, project env:
source ~/env.sh
python scripts/math500_uniform_k16.py --out-root /path/to/math-rollouts-data

# then score on CPU (default scorer reproduces the legacy openings_k16 is_correct):
math-rollouts-score \
  --rollouts /path/to/math-rollouts-data/generations/qwen2.5-math-1.5b/math500_uniform_k16_d1/rollouts.parquet
```

The generic entry point is `math-rollouts-generate` (see `--help`).

## Loading data

```python
from math_rollouts.data.hf import load_scored_rollouts
df = load_scored_rollouts("Qwen/Qwen2.5-Math-1.5B", "math500_uniform_k16_d1")
```

Point at a local snapshot with `MATH_ROLLOUTS_DATA=/path/to/dataset`, otherwise
files are fetched from the HF hub and cached.

## Tests

```bash
pytest tests/      # schema dtypes, adapter wiring, depth-1 nucleus parity
```
