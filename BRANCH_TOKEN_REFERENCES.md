# Branch tokens — prior-work findings (reference)

> **Status:** reference notes for our core research question, not a spec. We are
> studying **branch tokens** — their frequency, size, and ability to predict
> correctness. This doc collects what prior work has already found about
> branch-/reflection-like behaviors, so our hypotheses are framed against known
> results rather than from scratch. Companion to [SCORING_REFERENCES.md](SCORING_REFERENCES.md)
> (which covers the orthogonal scoring/sampling question). Currently sourced from the
> Dr. GRPO paper; extend as we read more.

## 0. Why this is separate from the scoring doc

Scoring is "what counts as correct." This is "do branch behaviors *explain*
correctness." The Dr. GRPO paper happens to run a close analog of our central
experiment for **self-reflection** (a branch-like behavior), and the result is a
useful caution. Self-reflection ≠ our branch tokens, but it's the nearest published
precedent and the methodology transfers.

## 1. Dr. GRPO — self-reflection as a branch-like behavior

- **Paper (local):** [Dr_GRPO.md](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md)
- **Code (local):** [guided-rollouts/external/understand-r1-zero](../guided-rollouts/external/understand-r1-zero) @ `dfca49d`

### 1a. ⭐ Self-reflection does *not* reliably predict inference-time correctness
The headline caution for us. On DeepSeek-R1-Zero, for each question that elicited
≥1 reflecting response, they sampled **100 responses**, split them into
*has-reflection* vs *no-reflection*, and computed the per-question accuracy delta
([:565+](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L565)):

> *"nearly half [of] responses with self-reflection do not achieve higher accuracy
> than those without."*

Important scoping: this is an **inference-stage correlation** claim. They explicitly
allow that reflection may still aid **exploration during training** — a separate,
out-of-scope effect ([:155](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L155)).
→ For us: "branch tokens predict correctness" should not be assumed; the naive
version already failed for one branch-like behavior. Design the analysis to *measure*
the correlation (with a no-branch control group), not presuppose it.

### 1b. Detection methodology (transferable)
A two-stage detector with a precision-first bias
([:521-534](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L521)):

- **Keyword stage** — a deliberately *small, high-precision* pool:
  `recheck, rethink, reassess, reevaluate, re-evaluate, re-examine, reexamine,
  reconsider, reanalyze, double-check, check again, think again, verify again,
  go over the steps`. They **explicitly exclude** `"wait"` and `"try again"` as
  too false-positive-prone — a relevant warning if our branch-token detector keys on
  surface cues.
- **LLM stage** — GPT-4o-mini to catch *implicit* reflection (no keyword) and to
  filter keyword false positives; the two are **cross-validated** against each other
  for robustness.

### 1c. Branch-keyword usage is model-family-specific
([:525](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L525)) Qwen2.5 favors
`check again / double-check / re-evaluate / re-examine / recheck / reconsider / verify
again`; the DeepSeek family *never* uses `re-evaluate / re-examine / verify again`;
Llama leans on `think again`. They attribute it to pretraining-data differences.
→ For us: a branch-token vocabulary tuned on one model family may not transfer; calibrate
per base model.

### 1d. Branch behaviors are present in *base* models, pre-RL
The "Aha moment" (self-reflection keywords like "Aha"/"wait") already appears in
**DeepSeek-V3-Base** before any RL
([:144-155](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L144)), and RL
*amplifies frequency* rather than *introducing* the behavior. → Branch frequency is
partly a pretraining artifact; comparing base vs R'd models on branch rate is a
meaningful axis, but don't read presence as RL-induced.

### 1e. Length is a confounded predictor of correctness
Incorrect responses are *notably longer* than correct ones, but the paper attributes
this to a **difficulty confound** (harder questions → longer responses *and* more
errors), not a causal length→wrong link
([:558](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L558)). Separately, the
whole Dr. GRPO thesis is that GRPO's optimization bias *inflates length specifically on
incorrect responses* ([:216](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L216)).
→ For our "branch size" axis: control for question difficulty before reading any
length/size → correctness signal, and remember the size distribution itself is shaped by
the training algorithm.

## 2. Open implications for our study *(to refine)*

- Treat correctness-prediction as a hypothesis to **test against a control**, per §1a.
- Reuse the precision-first detector shape (§1b); be wary of `wait`-class tokens.
- Calibrate branch vocabulary per model family (§1c) and report base-vs-RL branch rates
  knowing the base already branches (§1d).
- Difficulty-stratify before any size↔correctness claim (§1e).

## 3. Provenance

| What | Path | Rev |
|---|---|---|
| Dr. GRPO paper (md) | `guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md` | — |
| Dr. GRPO code | `guided-rollouts/external/understand-r1-zero` | `dfca49d` |

> Next candidates to mine for branch-relevant findings: Limits-of-RLVR (coverage /
> reasoning-path analysis) and any teacher-guidance prior work, as we get to them.
