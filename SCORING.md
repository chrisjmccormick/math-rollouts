# Scoring methodology (PROPOSAL — for review)

> Status: draft for sign-off. Replaces the single baked-in `is_correct` column with
> (a) **raw, criterion-free attributes** on every rollout and (b) **named, documented
> scorers** that apply a policy over those attributes. Benchmark numbers are always
> reproduced from a named scorer + params — never read off a stored boolean.
> Grounded in [SCORING_REFERENCES.md](SCORING_REFERENCES.md) (Dr. GRPO, Limits-of-RLVR).

## 1. Why `is_correct` is going away (and the band-shift is resolved)

The legacy pools set `is_correct = math_verify(full completion)` — a permissive,
**truncation-tolerant** match (`run_random_nothink.py`; no box gate, no termination
gate). The canonical `boxed-match-stop-v1` added `require_stop=True`, which moved
10.6% of base-model MATH-500 problems across difficulty bands. But **neither
reference paper penalizes truncation** (Dr. GRPO scores a truncated-but-boxed
rollout correct; Limits-of-RLVR has no termination gating). Adopting their
truncation-tolerant default reproduces the legacy verdict **exactly** — re-scored
drift is **0 flips / 0 bands moved** on base `math500_passK` and `math12k_additional`.

So the fix is not to pick a "better boolean" — it's to stop storing one. We store
the *facts*; difficulty bands and accuracy are defined by an explicitly **named
scorer** (default: truncation-tolerant), so any future change is a visible
parameter, not a silent re-score.

## 2. Raw attributes (one row per rollout, criterion-free)

Generation facts (unchanged): `model_id, unique_id, subject, answer, run_id,
gen_config_id, seed, temperature, top_p, sample_idx, completion_token_ids,
completion_text`.

**Termination** — keep the industry-standard vLLM/OpenAI fields verbatim, plus one
derived label:

| column | type | meaning |
|---|---|---|
| `finish_reason` | str | raw vLLM category: `stop` / `length` / `abort` / `error` / `repetition` |
| `stop_reason` | str\|int\|null | `null`=natural EOS; str=matched stop-string; int=matched stop-token-id; `"repetition_detected"` |
| `terminal` | enum | derived: `emitted_eos` / `stop_string` / `truncated` / `repetition` / `aborted` / `error` |

Derivation: `length`→`truncated`; `stop`+`stop_reason is None`→`emitted_eos`;
`stop`+non-null→`stop_string`; `repetition`/`abort`/`error` pass through.

> **vLLM EOS semantics (resolved).** `finish_reason` is a *category*, never a token.
> The EOS token **is** returned as the final entry of `completion_token_ids`
> (`<|endoftext|>`=151643 for Qwen2.5-Math) but is stripped from `completion_text`
> (`skip_special_tokens`). `<|im_end|>`=151645 is the ChatML turn-marker used by
> *instruct* variants; base-math models end on `<|endoftext|>`, so the generator's
> defensive `stop=["<|im_end|>"]` never fires (`stop_reason is None`,
> `terminal==emitted_eos`).
>
> **`max_tokens` is an output budget that INCLUDES the EOS token** (verified: at
> cap=3000, every `truncated` row is exactly 3000 tokens with no EOS; every
> `emitted_eos` row is ≤3000 with the trailing 151643 counted). So a natural
> completion has ≤3000 tokens *including* EOS; a truncated one has exactly 3000
> content tokens.

**Lengths** (EOS token excluded from the response count so termination types are
comparable):

| column | meaning |
|---|---|
| `completion_num_tokens` | response length, **excluding** the trailing EOS token |
| `prompt_num_tokens` | tokens in the rendered prompt |
| `total_num_tokens` | `prompt + completion` (sequence length, EOS-excluded) |
| `max_gen_len` | the generation budget this rollout was produced under (load-bearing) |

**Answer / match attributes** (`math_verify` + `analysis.positional`):

| column | meaning |
|---|---|
| `answer_matches` | bool — permissive `math_verify` over the full completion (no box gate, no termination gate). **Replaces `is_correct`.** == legacy semantics. |
| `has_boxed` | bool — a closing `\boxed{…}` is present (enables a Dr. GRPO-style box-gated scorer) |
| `answer_char_pos` | char offset where the verified answer first appears, or null |
| `answer_token_frac` | that position as a fraction of the response (tokens), or null |

> `is_correct` is **dropped outright** (no deprecated alias) — consumers move to a
> named scorer.

## 3. Scorers (named policies over the raw attributes)

Pure, reproducible, never stored on raw rows (cached in analysis tables keyed by
`scorer_id`). Maps a rollout → `{correct, incorrect, unresolved}`.

### `answer-match` — **DEFAULT for reporting** (truncation-tolerant)
`correct ⟺ answer_matches`. No box gate, no termination gate. Matches
Limits-of-RLVR and the legacy pools; appropriate for the branch-token analysis
(permissive by design). **This is what difficulty bands / headline accuracy use.**

### `boxed-match` (Dr. GRPO-style, stricter)
`correct ⟺ has_boxed ∧ answer_matches`. Requires the answer to be boxed (the
"biggest lever" between the two references). For a benchmark-grade view.

### `benchmark@budget=B` (budget-aware; for length-controlled comparisons)
`answer_matches ∧ terminal==emitted_eos` → correct; `terminal==truncated ∧
max_gen_len < B` → **unresolved**; else incorrect. **Strict mode** (default when a
single number is requested) **raises** on any `unresolved` — the pool must be
regenerated at ≥ B first. Use when comparing models under a fixed token budget.

> Worked example: Qwen3-8B rollouts at `max_gen_len=8192`, budget `B=10240` →
> truncated rows are `unresolved` and strict scoring errors ("extend to 10240
> first"). At `max_gen_len=10240 ≥ B`, the same truncated row is a clean `incorrect`.

### `leak-filtered@keep_frac=0.70`
`answer-match ∧ answer_token_frac ≥ keep_frac` → `keeper`; correct-but-early →
`leak`. Reuses `analysis.positional`.

## 4. Migration impact

- Re-derive raw attributes for all 7 pools once. `is_correct`→`answer_matches`; add
  `has_boxed`, termination (`finish_reason`/`stop_reason`/`terminal`), lengths,
  placement.
- Under the **default `answer-match`** scorer, base-model bands are **unchanged
  (0 drift)** — published numbers preserved.
- `*_token_nuclei` shards copy `answer_matches` (the default verdict) instead of
  `is_correct`; refreshed by a join, as today.
- The tighter teacher-guidance work can adopt `boxed-match` or `benchmark@budget`
  later without re-touching the raw data.
