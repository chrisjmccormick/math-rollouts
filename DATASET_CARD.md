---
license: mit
pretty_name: math-rollouts
language:
  - en
task_categories:
  - text-generation
tags:
  - math
  - reasoning
  - rollouts
  - nucleus-sampling
  - qwen2.5-math
  - math-500
size_categories:
  - 100K<n<1M
---

# math-rollouts

Naturally-sampled math-reasoning **rollouts** for **Qwen2.5-Math-1.5B** (base) and
**Qwen2.5-Math-1.5B-Oat-Zero**, plus first-token / branch **nuclei** and
uniform-opener forced rollouts. Generation is guidance-free: only naturally-sampled
completions from public checkpoints.

- **Code (generator, adapters, nucleus tree, scorers, analysis):**
  [github.com/chrisjmccormick/math-rollouts](https://github.com/chrisjmccormick/math-rollouts)
- **Data (this repo, `repo_type=dataset`):** the parquets/CSVs documented below.

The code/data split keeps git light, lets consumers (e.g. the `nucleus-viz` repo)
pull parquets independently, and keeps generation reproducible from code.

## Layout

```
problems/
  math500.parquet            the 500 math500-split problems (+ hf MATH-500 cross-ref)
  math_problems.parquet      the full ~12.5k MATH superset ("math12k"), with `split`
mappings/
  math500_to_hf.csv          unique_id  <->  hf MATH-500 id (test/<subj>/<n>.json)
generations/<model-slug>/
  <experiment>/              the unified-generator split (nuclei + raw rollouts + scores)
    nuclei.parquet
    rollouts.parquet
    scores.parquet
    policies.csv
    manifest.json
  <pool>.parquet             self-contained naturally-sampled pools (is_correct inline)
```

`model-slug` = lowercase HF id minus the org (`Qwen/Qwen2.5-Math-1.5B` ->
`qwen2.5-math-1.5b`; `sail/Qwen2.5-Math-1.5B-Oat-Zero` -> `qwen2.5-math-1.5b-oat-zero`).

## Ids and splits

Every row has **one** problem id, `unique_id = <split>/<subj>/<n>`, where `split` is
one of **`train`** / **`test`** / **`math500`** and `<n>` is the stable math12k
number. The split is authoritative (also a column in `math_problems.parquet`):

| split | count | meaning |
|---|---|---|
| `train` | 7,496 | math12k train portion |
| `test`  | 4,500 | math12k test portion (excluding math500) |
| `math500` | 500 | the MATH-500 subset, **held out of `test`** |

So `math500/geometry/9467` is the MATH-500 geometry problem that lives at math12k
index 9467. The same id is used everywhere — pools, experiment files, and
`policies.csv` (older files keyed math500 problems by the HF id `test/.../<n>.json`;
that's gone). To recover the canonical HF MATH-500 id (e.g. `test/geometry/627.json`,
for matching published numbers), join `mappings/math500_to_hf.csv`. There is no
longer a second id column on the bulk rows.

## Models

| model-slug | checkpoint | notes |
|---|---|---|
| `qwen2.5-math-1.5b` | `Qwen/Qwen2.5-Math-1.5B` | base; broad first-token nuclei |
| `qwen2.5-math-1.5b-oat-zero` | `sail/Qwen2.5-Math-1.5B-Oat-Zero` | RL-tuned; first-token distribution is sharply peaked, so most problems yield a **singleton** nucleus (one opener) — `math500_uniform_k16_d1` has 553 openers for 500 problems vs the base model's 3,275 |

## Generation config

All canonical runs share one sampling config (`gen_config_id = 200`), used for **both**
the nucleus fan-out and the forced/natural rollouts:

```
temperature = 0.6   top_p = 0.95   top_k = 20   max_tokens = 3000   max_model_len = 4096
```

`top_k` caps the nucleus size; the nucleus is computed on temperature-scaled probs,
kept by top-p (always keeping the top token), then renormalized within the kept set.
Rows with a different `gen_config_id` are **not** the same sampling distribution — do
not pool across them.

> **Precision matters.** Rollouts were sampled with **bfloat16** (vLLM). Some
> first-token logits are nearly tied, so recomputing nucleus membership/probabilities
> in fp32 can reshuffle the nucleus. Match bf16 when recomputing per-token quantities.

## File families & schemas

### `problems/`

`math500.parquet` (500 rows): `unique_id` (`math500/...`), `math500_native_id` (the
hf MATH-500 cross-ref, kept here), `subject`, `subj`, `level`, `problem`, `solution`,
`answer`.
`math_problems.parquet` (12,496 rows, the math12k superset): `unique_id`
(`<split>/...`), `source_idx`, `split`, `subject`, `subj`, `level`, `problem`,
`solution`, `answer`.

### Experiment split — `generations/<model>/<experiment>/`

Currently: `math500_uniform_k16_d1` (depth-1 first-token nucleus, K=16 forced uniform
rollouts per opener, all 500 MATH-500 problems) for both models. **Generation and
scoring are separate**: `rollouts.parquet` is RAW (no correctness); `scores.parquet`
is derived under a versioned `scorer_id` and re-runnable on CPU.

**`nuclei.parquet`** — one row per OPENER (= leaf of the nucleus tree):
`model_id`, `unique_id`, `subject`, `answer`, `depth`,
`branch_path` (child-index at each fork, root->leaf — the durable opener identity),
`opener_token_ids`, `opener_token_strs`, `fork_token_id`, `nuc_prob` (renormalized
inbound prob), `path_prob`, `branch_size`, `terminal`, `is_thinking`.

**`rollouts.parquet`** — RAW forced samples, one row per sample (no correctness):
`model_id`, `unique_id`, `subject`, `answer`, `depth`,
`branch_path`, `opener_token_ids`, `run_id`, `gen_config_id`, `seed`, `temperature`,
`top_p`, `max_gen_len`, `sample_idx`, `completion_token_ids`, `completion_text`
(includes the forced opener), `num_tokens`, `finish_reason`.

**`scores.parquet`** — derived, one row per (rollout × scorer):
`model_id`, `unique_id`, `run_id`, `branch_path`, `sample_idx`, `scorer_id`
(e.g. `boxed-match-stop-v1`), `is_correct`, `answer_char_pos`, `answer_token_frac`,
`leak_class`. Join back to rollouts on
`(model_id, unique_id, run_id, branch_path, sample_idx)`.

**`policies.csv`** — per-problem opener-policy accuracy summary: `unique_id`, `subject`,
`n_openers`, `probability`, `uniform`, `acc_weighted`, `oracle`.

**`manifest.json`** — `model_id`, `gen_config`, `gen_config_id`, `k`, `max_depth`,
`max_branch`, `run_id`, `seed`, `coverage`, `n_problems`, `n_openers`, `n_rollouts`,
`created_utc`.

### Naturally-sampled pools — `generations/<model>/<pool>.parquet`

Self-contained, **`is_correct` inline** (no separate scores file), one row per sample.
These are NOT opener-forced — the model chose its own first token. Pools include
`math500_passK`, `math12k_passK`, `math12k_K64`, `math12k_L4_5_K64`, `math12k_additional`
(availability varies by model). Columns:

`unique_id` (`<split>/...`), `problem_idx`, `run_id`, `sample_idx`, `producer`
(e.g. `random`), `completion_text`, `completion_token_ids`, `num_tokens`,
`initial_num_tokens`, `is_correct`, `finish_reason`, `closed_think`, `n_inside_think`,
`n_post_think`, `seed`, `temperature`, `top_p`, `model_id`, `max_gen_len`,
`gen_config_id`, `timestamp`, `subject`, `level`, `answer`. The `*_think` columns are generic carry-overs and are inert for
these non-thinking models.

## Grouping & accuracy

The "these K were generated together" key for the experiment split is
`(model_id, unique_id, branch_path, run_id)`, and
`accuracy = sum(is_correct) / group_size` where `group_size` is the **row count** for
that key (never a stored count). For the pools, group by `(model_id, unique_id)` (and
`run_id` if pooling deliberately). `branch_path` — child-index at each fork — is the
durable opener identity, since a raw fork token id can recur across forks at depth > 1.

## Loading

With the [`math-rollouts`](https://github.com/chrisjmccormick/math-rollouts) package
(`pip install -e .`):

```python
from math_rollouts.data.hf import (
    load_scored_rollouts, load_nuclei, load_generation_parquet,
)

# experiment split: raw rollouts joined to their scores
df = load_scored_rollouts("Qwen/Qwen2.5-Math-1.5B", "math500_uniform_k16_d1")

# nuclei (one row per opener)
nuc = load_nuclei("Qwen/Qwen2.5-Math-1.5B", "math500_uniform_k16_d1")

# a naturally-sampled pool, one MATH-500 problem (= hf test/geometry/627.json)
pool = load_generation_parquet("Qwen/Qwen2.5-Math-1.5B", "math500_passK")
geom = pool[pool.unique_id == "math500/geometry/9467"]
```

Point at a local snapshot with `MATH_ROLLOUTS_DATA=/path/to/dataset`; otherwise files
are fetched from the hub and cached. Plain pandas works too:

```python
import pandas as pd
from huggingface_hub import hf_hub_download
p = hf_hub_download("ChrisMcCormick/math-rollouts",
                    "generations/qwen2.5-math-1.5b/math500_passK.parquet",
                    repo_type="dataset")
df = pd.read_parquet(p)
```

## How this dataset was built

**Problems (`problems/`).** The math12k pool is the
[`qwedsacf/competition_math`](https://huggingface.co/datasets/qwedsacf/competition_math)
mirror of Hendrycks MATH (12,500 rows, train+test bundled with no split labels). Each
row gets a stable `unique_id = <split>/<subj>/<source_idx>`, where `source_idx` is its
row index in that dataset (so the id is reproducible), and:

- **`math500`** — the row's problem text normalized-matches a problem in
  [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
  (exactly 500 matches); held out as its own split.
- **`test`** — `source_idx >= 7500` (and not math500).
- **`train`** — `source_idx < 7500`.

`problems/math500.parquet` is the 500 math500-split rows plus their HF MATH-500
`math500_native_id` (the same cross-ref as `mappings/math500_to_hf.csv`). Rows with
no boxed answer (~4) are dropped.

> The original `unique_id`s used a `train/` prefix for every row and a separate
> `math500_native_id` (with the HF MATH-500 set as the test authority); since the
> overlapping problem text + answers are identical, that was collapsed to the single
> split-aware id above and the HF set demoted to a cross-ref.

**Rollouts (`generations/`).** Generated from the public **Qwen2.5-Math-1.5B** and
**Qwen2.5-Math-1.5B-Oat-Zero** checkpoints with the config above (the `math-rollouts`
generator: nucleus pass in HF + forced/natural sampling in vLLM, bf16). Scoring is a
separate CPU pass. Excluded by design: in-house fine-tunes and teacher-guided
(intersection-sampling) rollouts.
