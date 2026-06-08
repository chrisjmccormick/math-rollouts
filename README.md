# math-rollouts

Naturally-sampled reasoning rollouts on the **MATH** benchmark — the ~12.5k-problem
"math12k" pool and the held-out **MATH-500** subset — from public **Qwen2.5-Math-1.5B**
checkpoints (the base model and the RL-tuned **Oat-Zero**), plus first-token / branch
**nuclei** and per-token nucleus statistics.

It's a reusable substrate for studying **how RL tuning reshapes reasoning**: the same
problems sampled many times (K rollouts each) from base vs. tuned models, with the
token-level structure (nuclei, branches) captured alongside the completions. A unified
generator produces the rollouts; a separate, re-runnable pass scores them.

**Code lives here; data lives in the HF dataset
[`ChrisMcCormick/math-rollouts`](https://huggingface.co/datasets/ChrisMcCormick/math-rollouts)**
(`repo_type=dataset`) — see its card for the full schema. The split keeps git light and
lets consumers (e.g. the separate [`nucleus-viz`](https://github.com/chrisjmccormick/nucleus-viz)
repo, which does the fancy visual analysis) pull parquets independently.

## What you can study with it

- **Difficulty banding** — re-rank problems by the base model's *empirical* solve rate
  (vs. MATH's static L1–L5) to see where RL helps and where it regresses.
- **Opener effects** — uniform-opener rollouts (sample the first token uniformly from
  its nucleus, then force the continuation) measure how much the opening branch
  determines the outcome. It matters a surprising amount.
- **Reachability / pass@k** — base vs. Oat-Zero pass@k crossovers: RL lifts low-k
  accuracy but can close off access to solutions on the hardest problems.
- **Open / closed branches** — under standard sampling, is each token of one model's
  rollout inside the *other* model's nucleus at that position? Tokens that fall outside
  mark branches the tuning **closed** (or, in reverse, **opened**). The per-token
  nucleus store (below) is the substrate for this.

## What's where

| | |
|---|---|
| **Code (this repo)** | unified generator, model adapters, nucleus tree, per-token nucleus trace/stats, scorers, analysis (accuracy/policies, difficulty banding) |
| **Data (HF dataset)** | `problems/`, `mappings/`, `generations/<model-slug>/<experiment-or-pool>` parquets + manifests — see the dataset card |

Problem identity is a single split-aware `unique_id = <split>/<subj>/<n>` with
`split ∈ {train, test, math500}` (`math500` held out of `test`); the dataset's own
`problems/math_problems.parquet` is the authority — no external MATH-500 dependency.

## Core ideas

- **One code path for nuclei.** A first-token nucleus is a depth-1 branch tree;
  `NucleusTree(max_depth=1)` reproduces the legacy `openings_k16` recipe byte-for-byte,
  and `max_depth>1` walks deeper with one persistent KV cache. The same recipe
  (`nucleus/recipe.py`) is reused by the per-token trace (`nucleus/trace.py`) and the
  pool-wide nucleus statistics (`analysis/token_nuclei.py`).
- **Model differences live in one ABC.** `adapters/base.py:ModelAdapter` captures the
  only thinking/non-thinking differences (prompt, terminals, scoring, vLLM stops).
  Adding a model = one subclass + one registry line — no schema or generator change.
- **Generation and scoring are separated.** Generation writes pristine RAW rollouts
  (GPU); scoring (`score/run.py`, CPU) writes `scores.parquet` under a versioned
  `scorer_id`, so accuracy can be recomputed under a different scorer without
  regenerating.
- **Grouping is explicit.** The group key is `(model_id, unique_id, branch_path,
  run_id)`; `accuracy = Σ is_correct / group_size` (row count, never a stored count).
  `branch_path` (child-index at each fork) is the durable opener identity.

## Install

```bash
pip install -e .            # CPU: data loading, scoring, analysis, nucleus stats
pip install -e '.[gen]'     # + torch/transformers/vllm for GPU generation
pip install -e '.[dev]'     # + pytest
```

## Usage

The generation, scoring, and analysis entry points are installed as console scripts
(and are importable as `math_rollouts.*` functions). `scripts/` holds the one named
**job recipe** that isn't a generic CLI.

```bash
# Generate (GPU): nucleus pass (HF) -> forced rollouts (vLLM)
math-rollouts-generate --model Qwen/Qwen2.5-Math-1.5B \
    --experiment math500_uniform_k16_d1 --k 16 --max-depth 1 --out-root <data-root>
# ...or the canonical first-job recipe, with its parameters baked in:
python scripts/math500_uniform_k16.py --out-root <data-root>

# Score (CPU, re-runnable under any scorer_id)
math-rollouts-score --rollouts <data-root>/generations/<slug>/<exp>/rollouts.parquet

# Analysis
math-rollouts-policies   --exp-dir <data-root>/generations/<slug>/<exp>   # opener-policy accuracy
math-rollouts-bandtable  ...                                              # difficulty-banded comparison
math-rollouts-token-nuclei --pool math500_passK --out-root <data-root>    # per-token nucleus sizes + store
```

GPU phases must run in the project env (`source ~/env.sh`).

**Notebook.** `notebooks/calculate_token_nuclei.md` runs the per-token nucleus job on a
Colab GPU and pushes the store back to the dataset (it's a markdown-form notebook;
convert it with your `.md`↔`.ipynb` utility, or run it as a plain script).

## Loading data

```python
from math_rollouts.data.hf import load_scored_rollouts, load_generation_parquet

# experiment split: raw rollouts joined to their scores
df = load_scored_rollouts("Qwen/Qwen2.5-Math-1.5B", "math500_uniform_k16_d1")

# a naturally-sampled pool, one problem
pool = load_generation_parquet("Qwen/Qwen2.5-Math-1.5B", "math500_passK")
geom = pool[pool.unique_id == "math500/geometry/9467"]
```

Point at a local snapshot with `MATH_ROLLOUTS_DATA=/path/to/dataset`; otherwise files
are fetched from the hub and cached.

## Provenance & migrations

The base dataset was built once (recipe documented in the dataset card). The data has
since been re-keyed to the split-aware id scheme — `scripts/migrate_unique_id_splits.py`
records that transformation.

## Tests

```bash
pytest tests/      # schema dtypes, adapter wiring, nucleus/trace/stats parity
```
