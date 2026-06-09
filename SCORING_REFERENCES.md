# Scoring & sampling — reference methodology from prior work

> **Status:** reference notes, not a spec. This documents *what two papers we lean
> on actually did* for MATH500-style generation and answer scoring, at the code
> level, so our own choices in [SCORING.md](SCORING.md) can be made against a
> known baseline. It deliberately does **not** prescribe our approach — that's for
> us + the implementation agent to decide and then record back here under
> §6 "Our approach vs. these references".

## 0. Why we reference these papers (and where our needs differ)

We are **not** reproducing these papers or competing on their numbers. We're doing
our own analysis — primarily **studying branch tokens** (their frequency, size, and
ability to predict correctness) — and we want defensible, externally-grounded
standards for the supporting pieces (sampling config, what "correct" means).

Two things shape how we read these references:

- **We use `<think>` models.** Qwen2.5-Math (the model both papers center on) is
  **not** a thinking model, so neither paper's main MATH500 path addresses *where*
  to extract an answer relative to a `</think>` boundary. That gap is ours to fill
  (cf. our `post-think-v1` scorer).
- **Our "correctness" leans permissive — by design.** Because we're *analyzing*
  rather than *competing*, a lenient correctness signal is appropriate for the
  branch-token study. The follow-on **teacher-guidance** work may need a tighter,
  benchmark-grade notion of correctness; we expect to define that separately rather
  than retrofit the analysis metric.

So treat the two references as endpoints of a spectrum we're positioning within,
not as targets.

## 1. The permissiveness spectrum (orientation)

| Axis | Dr. GRPO / Oat-Zero | Limits-of-RLVR | (our `check_correct` today) |
|---|---|---|---|
| Requires `\boxed{}`? | **Yes — hard gate** | No (boxed → "answer is" → last number) | No (math_verify over full text) |
| Answer equivalence | mathd-norm ∨ sympy ∨ math_verify | sympy `math_equal` | math_verify `verify` |
| Truncation penalized? | No | No | Only under `*-stop-v1` scorers |
| Decoding | T=0, pass@1 | T=0.6 / top-p 0.95, pass@k | (ours) |
| Gen budget | 3000 tok | 16,384 tok | (ours) |
| Model sizes studied | 1.5B **and** 7B | **7B-centric** (no 1.5B) | (ours: 1.5B + `<think>`) |

The two references bracket the box question: Dr. GRPO is the strict end (no box →
wrong), Limits-of-RLVR is the lenient end (no box required at all). Both are
permissive about *equivalence* once an answer is extracted.

---

## 2. Dr. GRPO / "Understanding R1-Zero-Like Training" (Oat-Zero)

- **Paper (local):** [Dr_GRPO.md](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md)
- **Code (local clone):** [guided-rollouts/external/understand-r1-zero](../guided-rollouts/external/understand-r1-zero) — `sail-sg/understand-r1-zero` @ `dfca49d`
- **Models released:** `sail/Qwen2.5-Math-{1.5B,7B}-Oat-Zero` (both sizes — see
  [README.md:175-176](../guided-rollouts/external/understand-r1-zero/README.md#L175)).

### Sampling (MATH500 = the `math` task)
From [evaluate_model.py](../guided-rollouts/external/understand-r1-zero/evaluate_model.py):
- Template `qwen_math` → reward fn `boxed_reward_fn`. Prompt instructs *"put your
  final answer within `\boxed{}`."*
- **`temperature=0`, `top_p=1`, `n_samples=1` → T=0 pass@1.**
- `max_tokens=3000`, `max_model_len=4096`. Stop tokens are **commented out**, so
  generation runs to EOS or the 3000-token cap.
- Scored with `fast=False`, which enables the math_verify recall path.

### Answer scoring — `boxed_reward_fn`
From [math_grader.py](../guided-rollouts/external/understand-r1-zero/understand_r1_zero/math_grader.py):
- **`\boxed{}` is a hard requirement.** `extract_answer` returns `None` if `\boxed`
  is absent → reward `0.0` ("Cannot even parse anything"). No box ⇒ wrong, full stop.
- Extraction = last `\boxed{...}` (`rfind` + brace match). If truncation cuts the
  box before its closing `}`, `remove_boxed` fails → `None` → 0.
- Once extracted, equivalence is high-recall: `grade_answer_mathd` **OR**
  `grade_answer_sympy` **OR** (`fast=False`) `is_latex_equal`, the last of which runs
  HF `math_verify.verify(parse(...))` with `LatexExtractionConfig(boxed_match_priority=0)`
  + `ExprExtractionConfig`. **Crucially, math_verify only ever sees the
  already-extracted boxed content — it cannot rescue an un-boxed answer.**

### Two questions we asked of it
- **Truncated rollouts with a boxed answer → scored correct.** There is *no*
  truncation/`finish_reason` handling anywhere; token lengths are reported, never
  scored. A truncated-but-well-boxed rollout is graded on its merits.
- **Boxing is required by *their wrapper*, not by math_verify.** math_verify is the
  lenient comparator that runs *after* the box gate.

### 1.5B benchmark numbers & measurement sensitivities (MATH500)
The 7B AIME result is the paper's headline, but **the Dr. GRPO algorithm analysis
itself runs on Qwen2.5-Math-1.5B / Qwen2.5-1.5B** (method intro
[:282](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L282), template×data
study [:311](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L311), ablations
[:509](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L509)) — so the 1.5B
numbers are the load-bearing ones for us.

MATH500, greedy (T=0), 3k-token budget
([benchmark table :470-476](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L470)):

| 1.5B model | MATH500 |
|---|---|
| Qwen2.5-Math-1.5B, Qwen template | 33.0 |
| Qwen2.5-Math-1.5B, **no template** | 61.8 |
| **Oat-Zero-1.5B** | **74.2** |
| Qwen2.5-Math-1.5B-Instruct | 74.2 |
| R1-Distill-Qwen-1.5B @3k / @8k | 52.2 / 77.4 |

Two measurement sensitivities worth carrying into our config:
- **Prompt template swings the base-model score ~40 pts** (no-template 61.8 →
  R1-template 21.2 on the *same* Qwen2.5-Math-1.5B;
  [:110-127](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L110)). Verdict:
  *"applying templates in fact destroys the capability before RL reconstructs it"*
  ([:335](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L335)). The wrapper
  is a first-class scoring variable, not a detail.
- **Generation budget swings a `<think>`/distill model ~25 pts**: R1-Distill-Qwen-1.5B
  scores 52.2 @3k but 77.4 @8k
  ([:460](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L460),
  [:474-475](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L474)) — a 3k cap
  *halves* a thinking model's apparent MATH500. This is the strongest evidence in either
  reference for **budget-aware scoring** (cf. our `unresolved` notion).

### Two notions of "correct" within the paper
The **RL training reward** is the minimalist *"R=1 if the response contains the correct
final answer"* via Math-Verify, with **no format reward**
([:282-292](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L282)) — looser than
the box-gated `boxed_reward_fn` used at **evaluation**. So even within this one paper,
training-time and benchmark-time correctness are deliberately different notions — the same
analysis-vs-teacher-guidance split we're navigating.

### `<think>`-relevant note
The `r1` template path uses `answer_tag_reward_fn`, which is strict about format:
it requires `</think> <answer> … </answer>` and then extracts `\boxed{}` from inside
`<answer>`. (A looser `answer_tag_reward_fn_for_orz` exists for baselines.) This is
the closest thing in either reference to thinking-style answer extraction, but it's
a *format-tag* gate, not a `</think>`-region notion like ours.

---

## 3. Limits of RLVR — "Does RL Really Incentivize Reasoning Capacity… Beyond the Base Model?"

- **Paper (local):** [Limit_of_RLVR.md](../guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md) · project page https://limit-of-RLVR.github.io
- **Code (local clone):** [guided-rollouts/external/limit-of-RLVR](../guided-rollouts/external/limit-of-RLVR) — `LeapLabTHU/limit-of-RLVR` @ `c4c581d`
- **This is the source of our T=0.6 / top-p=0.95 decision.**

### Sampling
Paper [§RLVR for Math, :203](../guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md#L203):
- **Temperature 0.6, top-p 0.95, max generation 16,384 tokens**, applied identically
  to base and RLVR models.
- **Zero-shot** — deliberately no few-shot for the base model (avoids confounds);
  same prompt as RLVR training / benchmark default.
- Code defaults in [eval_math_nodes.sh:23-29](../guided-rollouts/external/limit-of-RLVR/math/eval_math_nodes.sh#L23)
  are placeholders (`temperature=0.0, top_p=1, max_tokens=16000`) overridden per run;
  the pass@k runs use `temp0.6` and the `qwen-boxed` prompt template (see filename
  pattern in [pass@k.py:51](../guided-rollouts/external/limit-of-RLVR/math/pass@k.py#L51)).
  **The paper text (0.95) is authoritative for the reported runs.**

### pass@k methodology (kept for reference; not our current focus)
Paper [Appendix "Low-Variance pass@k", :511-527](../guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md#L511):
- Unbiased low-variance estimator (Chen et al. 2021):
  `pass@k = E_x[ 1 − C(n−c, k) / C(n, k) ]`, generating `n ≥ k` samples per problem,
  `c` correct.
- `n` = the largest/rightmost `k` in each curve: **MATH500 → n=128** (also Minerva,
  GSM8K); AMC23/AIME24 → n=1024.
- **No "k=20 limiter" exists anywhere in this paper.** The only filtering is a
  heuristic *guessable-problem* pre-filter (prompt Qwen2.5-7B-Base to answer
  *directly, no CoT*; drop problems solvable with low-but-nonzero <5% probability),
  applied to **AIME24** (30→18 questions) as a robustness check —
  [:560](../guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md#L560) — **not** to MATH500's main pass@k.
- "Guessing" (wrong CoT, right answer) is otherwise handled by **manual CoT
  inspection** of the hardest problems (avg accuracy <5%), on GSM8K/AIME24 subsets.

### Answer scoring — Qwen2.5-Math `math_eval` harness
Code under [math/examples/math_eval/](../guided-rollouts/external/limit-of-RLVR/math/examples/math_eval):
- **`extract_answer`** ([parser.py:499](../guided-rollouts/external/limit-of-RLVR/math/examples/math_eval/parser.py#L499)):
  prefers `\boxed{}` (brace-matched) → else `"final answer is"` / `"the answer is"`
  → else **the last number** in the text. **No box required.**
- **Equivalence** = `math_equal` ([grader.py:61](../guided-rollouts/external/limit-of-RLVR/math/examples/math_eval/grader.py#L61))
  via `math_equal_process` with a 3-second timeout: string / numeric / symbolic
  (sympy) equality.
- **No termination/truncation gating** — the text is scored regardless of how it
  ended (and the 16k budget makes truncation rare anyway).

### Model coverage — **7B, not 1.5B**
The headline math analysis is **7B**: base Qwen2.5-Math-7B with SimpleRLZoo-7B,
**Oat-Zero-7B**, DAPO-32B, etc. The "model size scaling" section scales *up* toward
near-frontier (Magistral-Medium), **never down to 1.5B**
([:398-414](../guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md#L398)).
→ **There is no 1.5B pass@k curve in this paper to compare against.**

---

## 4. Consolidated facts that matter for our setting

- **1.5B vs 7B.** Any number quoted from *Limits-of-RLVR* (the "RL wins at low k,
  base overtakes at high k" story, the coverage tables, the Oat-Zero analysis) is
  **7B**. The only published **1.5B** Oat-Zero reference is **Dr. GRPO's own tables**
  — at **T=0, pass@1, box-gated grader**. So for our 1.5B work: Limits-of-RLVR gives
  the *qualitative shape* only; Dr. GRPO gives a pass@1 anchor (different metric and
  stricter grader than a T=0.6 pass@k sweep).
- **Box requirement is the biggest lever.** Dr. GRPO requires a box; Limits-of-RLVR
  does not. Our current `check_correct` does **not** require one (verified: it
  math_verify-parses the *whole* completion and takes the last expression, so
  "the answer is 42" / "= 42" / "x = 1/2" all score correct without a box). A
  permissive (no-box) stance matches Limits-of-RLVR and suits the analysis goal;
  the tighter teacher-guidance benchmark may want the Dr. GRPO-style gate.
- **Truncation.** Neither paper *penalizes* truncation in the scorer, but Dr. GRPO
  treats truncated responses as a **distinct, not-yet-scorable category** — it bins
  responses into correct/incorrect/unformatted/truncated and *excludes* truncated ones,
  noting they would "fall into any of the other three categories if a larger context size
  were used" ([:556-558](../guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md#L556)).
  That's external precedent for our `unresolved` notion. The budget gap matters: the
  references use 3000 / 16,384 tok, and the @3k→@8k jump above shows a too-short cap can
  halve a thinking model's score. So whether we gate on natural termination / budget is a
  real, visible choice — not a detail.
- **`<think>` extraction is unaddressed by both.** Whatever we do for the
  post-`</think>` region is our own extension; the nearest prior art is Dr. GRPO's
  `<answer>`-tag format gate, which is not the same thing.
- **Leak / guessing.** Our positional leak filter (answer must appear past 70% of the
  response) is a *proxy* for early-guess/echo, distinct from Limits-of-RLVR's manual
  CoT-validity inspection and its <5% direct-answer guessable-problem pre-filter.

## 5. Local provenance

| What | Path | Rev |
|---|---|---|
| Dr. GRPO paper (md) | `guided-rollouts/teacher-kv/paper/references/Dr_GRPO.md` | — |
| Dr. GRPO code | `guided-rollouts/external/understand-r1-zero` | `dfca49d` |
| Limits-of-RLVR paper (md) | `guided-rollouts/teacher-kv/paper/references/Limit_of_RLVR.md` | — |
| Limits-of-RLVR code | `guided-rollouts/external/limit-of-RLVR` | `c4c581d` |

## 6. Our approach vs. these references

Resolved choices (full spec in [SCORING.md](SCORING.md)):

- **Termination → not penalized by default.** Our default reporting scorer
  (`answer-match`) counts a rollout correct iff a correct answer is found, however
  it ended — matching Dr. GRPO's *reward fn* and Limits-of-RLVR, and reproducing the
  legacy pools **exactly** (re-score drift: **0 flips / 0 band moves** on base
  `math500_passK` and `math12k_additional`). The earlier 10.6% base-band shift was
  purely an artifact of a `require_stop` gate neither reference uses; it is dropped
  as the default.
- **…but truncation is preserved as a first-class axis.** Dr. GRPO's *reporting*
  bins truncated responses as a distinct, excluded category (§4) — external
  precedent for our `unresolved`. We capture that in a separate
  `benchmark@budget=B` scorer: `truncated ∧ max_gen_len < B → unresolved` (raises in
  strict mode). So "don't penalize truncation" (analysis default) and "truncated is
  not-yet-scorable below budget" (benchmark view) coexist as two named scorers, not
  one buried boolean. This matters most for **thinking models at short caps**, where
  the @3k→@8k effect (§4) can swing scores heavily; our 1.5B non-thinking pools
  truncate ~1% at 3000 tok, so the default is safe for the branch-token analysis.
- **Box requirement → permissive default, strict available.** `answer_matches` =
  `math_verify` over the full completion, **no box gate** (Limits-of-RLVR end; matches
  the existing `check_correct` and the analysis goal). We also store `has_boxed`, so
  the Dr. GRPO hard box gate is a one-column scorer (`boxed-match`) for a
  benchmark-grade view. The biggest lever is thus an explicit, named choice.
- **Equivalence → `math_verify.verify`** (high-recall, as both references use post-extraction).
- **Sampling → T=0.6 / top-p=0.95** (Limits-of-RLVR), `max_tokens=3000` /
  `max_model_len=4096` for the 1.5B Qwen-Math pools (matches Dr. GRPO's 1.5B budget).
  `max_tokens` **includes** the EOS token (verified).
- **Prompt template → per-model, set by the adapter registry**
  ([adapters/registry.py](src/math_rollouts/adapters/registry.py)), not a global flag:
  - `Qwen/Qwen2.5-Math-1.5B` (base) **and** `sail/Qwen2.5-Math-1.5B-Oat-Zero` →
    `QwenMathAdapter`: **the literal Qwen-Math template is applied** (boxed
    instruction in the system turn + ChatML, built as a raw string for byte-exact
    parity with the source `run_random_nothink` / `openings_k16` recipe).
  - `Qwen/Qwen3-8B` → `Qwen3ThinkAdapter` (its own `<think>` template);
    `Qwen/Qwen3-8B-Base` → `PaperBaseAdapter` (paper completion prompt
    `Question: … / Answer: Let's think step by step.`, no chat template).
  - **Deliberate trade-off for the 1.5B pair:** applying the Qwen-Math template to
    *both* base and Oat-Zero gives a **controlled, apples-to-apples** base-vs-Oat
    comparison (the right call for the branch-token study). The cost is that the
    **base** model is run at its *low-template* operating point — Dr. GRPO reports the
    same model at **33.0 (Qwen template) vs 61.8 (no template)** on MATH500 (§2), so
    these base numbers are not its ceiling. Where the base model's *true capability*
    matters, `paper_base`/no-template would be more representative.
- **`<think>` extraction → our own** `post-think-v1` (post-`</think>` region); the
  nearest prior art (Dr. GRPO's `<answer>`-tag gate) is a format gate, not a region.
- **Leak / guessing → positional filter** (`leak-filtered@keep_frac`, answer past 70%)
  as an early-guess/echo proxy — our extension.
- **Nucleus `top_k` cap → removed** (`top_k=-1`); sizes recorded uncapped, rare
  flat-distribution positions (nucleus in the hundreds) left pure for analysis to
  filter. (No reference analogue; noted for provenance.)
