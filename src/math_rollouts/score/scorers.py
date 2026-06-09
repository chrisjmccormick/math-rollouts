"""Named correctness scorers — POLICIES over criterion-free raw attributes.

The redesign stops storing a baked ``is_correct`` boolean. Generation/pool-building
records the *facts* about a rollout (``answer_matches``, ``has_boxed``, ``terminal``,
lengths, answer placement); a scorer is a pure, documented policy that maps those
facts to a verdict ``{correct, incorrect, unresolved}``. Benchmark numbers are always
reproduced from a named ``scorer_id`` + params, never read off a stored verdict.

Raw-attribute helpers (also used by ``data.pools`` when materializing a pool):
  ``answer_matches(text, answer)``  permissive math_verify over the FULL completion
                                    (no box gate, no termination gate; == legacy
                                    ``check_correct`` / ``boxed-match-v1`` semantics).
  ``has_boxed(text)``               a closing ``\\boxed{...}`` is present.
  ``derive_terminal(finish, stop)`` the derived termination enum.

Scorers (keyed by ``scorer_id``):
  ``answer-match``            DEFAULT. correct ⟺ answer_matches.
  ``boxed-match``             correct ⟺ has_boxed ∧ answer_matches (Dr. GRPO-style).
  ``benchmark@budget=B``      answer_matches ∧ terminal==emitted_eos → correct;
                              terminal==truncated ∧ max_gen_len<B → unresolved
                              (raises in strict mode); else incorrect.
  ``leak-filtered@keep_frac`` answer-match + positional leak class (keeper/leak).
  ``post-think-v1``           legacy: scores only the post-``</think>`` region.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..analysis.positional import KEEP_FRAC, classify, has_boxed, verified_answer_char_pos

THINK_CLOSE_STR = "</think>"

CORRECT, INCORRECT, UNRESOLVED = "correct", "incorrect", "unresolved"


def check_correct(completion: str, answer: str) -> bool:
    """math_verify on the full completion (mirrors the source check_correct). This is
    the criterion-free ``answer_matches`` fact: no box gate, no termination gate."""
    from math_verify import parse, verify

    gold = parse(f"\\boxed{{{answer}}}")
    try:
        return bool(verify(gold, parse(completion)))
    except Exception:
        return False


# ``answer_matches`` is the public name of the permissive full-text match fact.
answer_matches = check_correct


def check_correct_post_think(completion: str, answer: str) -> bool:
    """Correctness on the text AFTER the first ``</think>`` only (False if absent)."""
    idx = completion.find(THINK_CLOSE_STR)
    if idx == -1:
        return False
    return check_correct(completion[idx + len(THINK_CLOSE_STR):], answer)


def derive_terminal(finish_reason, stop_reason) -> str:
    """Derive the termination enum from the raw vLLM ``finish_reason`` +
    ``stop_reason``.  ``length``→``truncated``; ``stop``+null→``emitted_eos``;
    ``stop``+non-null→``stop_string``; repetition/abort/error pass through."""
    if stop_reason == "repetition_detected" or finish_reason == "repetition":
        return "repetition"
    if finish_reason == "length":
        return "truncated"
    if finish_reason == "stop":
        return "emitted_eos" if stop_reason is None else "stop_string"
    if finish_reason == "abort":
        return "aborted"
    if finish_reason == "error":
        return "error"
    return finish_reason or "error"


# --- reading raw attributes off a row (prefer stored facts; fall back to compute) ---

def _row_answer_matches(row: dict) -> bool:
    v = row.get("answer_matches")
    if v is None:
        v = check_correct(row["completion_text"], row["answer"])
    return bool(v)


def _row_has_boxed(row: dict) -> bool:
    v = row.get("has_boxed")
    if v is None:
        v = has_boxed(row["completion_text"])
    return bool(v)


def _row_terminal(row: dict) -> str:
    t = row.get("terminal")
    if t is None:
        t = derive_terminal(row.get("finish_reason"), row.get("stop_reason"))
    return t


def _keys(row: dict, scorer_id: str) -> dict:
    return {
        "model_id": row.get("model_id"), "unique_id": row.get("unique_id"),
        "run_id": row.get("run_id"), "branch_path": row.get("branch_path"),
        "sample_idx": row.get("sample_idx"), "scorer_id": scorer_id,
    }


def _result(row, scorer_id, verdict, *, answer_matches=None, has_boxed=None,
            char_pos=None, token_frac=None, leak_class=None) -> dict:
    return {
        **_keys(row, scorer_id),
        "verdict": verdict,
        "answer_matches": answer_matches,
        "has_boxed": has_boxed,
        "answer_char_pos": char_pos,
        "answer_token_frac": token_frac,
        "leak_class": leak_class,
    }


@dataclass
class AnswerMatchScorer:
    """DEFAULT reporting scorer: ``correct ⟺ answer_matches``. Truncation-tolerant,
    no box gate. Matches the legacy pools exactly (0 drift)."""

    scorer_id: str = "answer-match"

    def score_row(self, row: dict) -> dict:
        am = _row_answer_matches(row)
        return _result(row, self.scorer_id, CORRECT if am else INCORRECT,
                       answer_matches=am, has_boxed=_row_has_boxed(row))


@dataclass
class BoxedMatchScorer:
    """Dr. GRPO-style: ``correct ⟺ has_boxed ∧ answer_matches``."""

    scorer_id: str = "boxed-match"

    def score_row(self, row: dict) -> dict:
        am, hb = _row_answer_matches(row), _row_has_boxed(row)
        return _result(row, self.scorer_id, CORRECT if (am and hb) else INCORRECT,
                       answer_matches=am, has_boxed=hb)


@dataclass
class PostThinkScorer:
    """Legacy thinking-model scorer: ``answer_matches`` on the post-``</think>``
    region only (verdict feeds ``answer_matches`` so downstream analysis is uniform)."""

    scorer_id: str = "post-think-v1"

    def score_row(self, row: dict) -> dict:
        am = check_correct_post_think(row["completion_text"], row["answer"])
        return _result(row, self.scorer_id, CORRECT if am else INCORRECT,
                       answer_matches=am, has_boxed=_row_has_boxed(row))


@dataclass
class BenchmarkScorer:
    """Budget-aware, for length-controlled comparisons. ``answer_matches ∧
    terminal==emitted_eos`` → correct; ``terminal==truncated ∧ max_gen_len<budget``
    → unresolved; else incorrect. STRICT mode raises on ``unresolved`` (the pool must
    be regenerated at ≥ budget first)."""

    budget: int = 0
    strict: bool = True
    scorer_id: str = ""

    def __post_init__(self):
        if not self.scorer_id:
            self.scorer_id = f"benchmark@budget={self.budget}"

    def score_row(self, row: dict) -> dict:
        am = _row_answer_matches(row)
        terminal = _row_terminal(row)
        if am and terminal == "emitted_eos":
            verdict = CORRECT
        elif terminal == "truncated" and (row.get("max_gen_len") or 0) < self.budget:
            if self.strict:
                raise ValueError(
                    f"benchmark@budget={self.budget}: rollout truncated at "
                    f"max_gen_len={row.get('max_gen_len')} < {self.budget} is "
                    f"UNRESOLVED ({row.get('unique_id')}#{row.get('sample_idx')}); "
                    f"regenerate the pool at >= {self.budget} tokens first, or score "
                    f"in non-strict mode.")
            verdict = UNRESOLVED
        else:
            verdict = INCORRECT
        return _result(row, self.scorer_id, verdict,
                       answer_matches=am, has_boxed=_row_has_boxed(row))


@dataclass
class LeakFilterScorer:
    """``answer-match`` + positional leak classification. A correct rollout is a
    'keeper' only if the verified answer appears past ``keep_frac`` of the response;
    earlier is a 'leak'. Needs a tokenizer for the token fraction (falls back to a
    character fraction if none is supplied)."""

    keep_frac: float = KEEP_FRAC
    tokenizer: object = None
    scorer_id: str = ""

    def __post_init__(self):
        if not self.scorer_id:
            self.scorer_id = f"leak-filtered@keep_frac={self.keep_frac}"

    def score_row(self, row: dict) -> dict:
        am = _row_answer_matches(row)
        text, answer = row["completion_text"], row["answer"]
        # prefer stored positional facts; fall back to recompute.
        pos = row.get("answer_char_pos")
        frac = row.get("answer_token_frac")
        if pos is None and frac is None:
            pos = verified_answer_char_pos(text, answer)
            if pos is not None:
                if self.tokenizer is not None:
                    n_before = len(self.tokenizer.encode(text[:pos], add_special_tokens=False))
                    frac = n_before / max(len(row.get("completion_token_ids") or []) or len(text), 1)
                else:
                    frac = pos / max(len(text), 1)
        leak_class = classify(am, frac, self.keep_frac)
        return _result(row, self.scorer_id, CORRECT if am else INCORRECT,
                       answer_matches=am, has_boxed=_row_has_boxed(row),
                       char_pos=int(pos) if pos is not None else None,
                       token_frac=float(frac) if frac is not None else None,
                       leak_class=leak_class)


def _parse_params(rhs: str) -> dict:
    """``budget=10240`` / ``keep_frac=0.7`` -> ``{'budget': 10240}`` etc."""
    out = {}
    for part in rhs.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def get_scorer(scorer_id: str = "answer-match", *, strict: bool = True, **kw):
    """Resolve a ``scorer_id`` (with optional ``name@k=v,...`` params) to a scorer.

    Known names: ``answer-match`` (default), ``boxed-match``, ``benchmark@budget=B``,
    ``leak-filtered@keep_frac=F``, ``post-think-v1``."""
    name, _, rhs = scorer_id.partition("@")
    params = _parse_params(rhs) if rhs else {}

    if name == "answer-match":
        return AnswerMatchScorer()
    if name == "boxed-match":
        return BoxedMatchScorer()
    if name == "post-think-v1":
        return PostThinkScorer()
    if name == "benchmark":
        budget = int(params.get("budget", kw.get("budget", 0)))
        return BenchmarkScorer(budget=budget, strict=strict, scorer_id=scorer_id)
    if name == "leak-filtered":
        keep_frac = float(params.get("keep_frac", kw.get("keep_frac", KEEP_FRAC)))
        return LeakFilterScorer(keep_frac=keep_frac, tokenizer=kw.get("tokenizer"),
                                scorer_id=scorer_id)
    raise KeyError(
        f"unknown scorer {scorer_id!r}; known: answer-match, boxed-match, "
        f"benchmark@budget=B, leak-filtered@keep_frac=F, post-think-v1")
