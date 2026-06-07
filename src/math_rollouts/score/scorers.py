"""Versioned correctness scorers — the SEPARATE, re-runnable scoring pass.

Generation writes raw rollouts (pristine source of truth, no correctness columns).
Scoring reads those rows on CPU and emits ``scores.parquet`` rows. Each scorer has
a stable ``scorer_id`` so alternative scoring schemes coexist (additional rows /
sibling files) instead of forking the raw data.

Scorers implemented:
  ``boxed-match-stop-v1``  default; reproduces the legacy ``openings_k16`` scoring
                           (``is_correct = check_correct(text) and finish==stop``).
  ``boxed-match-v1``       ungated boxed match (ignores finish_reason).
  ``post-think-v1``        scores only the post-``</think>`` region (thinking models).
  ``leak-filter-v1``       boxed match + positional leak classification (keeper/leak).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..analysis.positional import KEEP_FRAC, classify, verified_answer_char_pos

THINK_CLOSE_STR = "</think>"


def check_correct(completion: str, answer: str) -> bool:
    """math_verify on the full completion (mirrors the source check_correct)."""
    from math_verify import parse, verify

    gold = parse(f"\\boxed{{{answer}}}")
    try:
        return bool(verify(gold, parse(completion)))
    except Exception:
        return False


def check_correct_post_think(completion: str, answer: str) -> bool:
    """Correctness on the text AFTER the first ``</think>`` only (False if absent)."""
    idx = completion.find(THINK_CLOSE_STR)
    if idx == -1:
        return False
    return check_correct(completion[idx + len(THINK_CLOSE_STR):], answer)


@dataclass
class Scorer:
    """Base scorer: boxed math_verify, optionally gated on ``finish_reason==stop``
    and/or scored on the post-``</think>`` region only."""

    scorer_id: str = "boxed-match-stop-v1"
    require_stop: bool = True
    post_think: bool = False

    def _verify(self, text: str, answer: str) -> bool:
        return check_correct_post_think(text, answer) if self.post_think else check_correct(text, answer)

    def score_row(self, row: dict) -> dict:
        text, answer = row["completion_text"], row["answer"]
        ok = self._verify(text, answer)
        if self.require_stop and row.get("finish_reason") != "stop":
            ok = False
        return {
            "model_id": row["model_id"], "unique_id": row["unique_id"],
            "run_id": row["run_id"], "branch_path": row["branch_path"],
            "sample_idx": row["sample_idx"], "scorer_id": self.scorer_id,
            "is_correct": bool(ok), "answer_char_pos": None,
            "answer_token_frac": None, "leak_class": None,
        }


class BoxedMatchScorer(Scorer):
    pass


@dataclass
class LeakFilterScorer(Scorer):
    """Boxed match + positional leak classification. A correct rollout is a
    'keeper' only if the verified answer appears past ``keep_frac`` of the
    response; earlier is a 'leak'. Needs a tokenizer for the token-fraction (falls
    back to a character fraction if none is supplied)."""

    scorer_id: str = "leak-filter-v1"
    keep_frac: float = KEEP_FRAC
    tokenizer: object = None

    def score_row(self, row: dict) -> dict:
        base = super().score_row(row)
        text, answer = row["completion_text"], row["answer"]
        pos = verified_answer_char_pos(text, answer)
        frac = None
        if pos is not None:
            if self.tokenizer is not None:
                n_before = len(self.tokenizer.encode(text[:pos], add_special_tokens=False))
                frac = n_before / max(len(row.get("completion_token_ids") or []) or len(text), 1)
            else:
                frac = pos / max(len(text), 1)
        base["answer_char_pos"] = int(pos) if pos is not None else None
        base["answer_token_frac"] = float(frac) if frac is not None else None
        base["leak_class"] = classify(base["is_correct"], frac, self.keep_frac)
        return base


_REGISTRY = {
    "boxed-match-stop-v1": lambda **kw: BoxedMatchScorer(scorer_id="boxed-match-stop-v1",
                                                         require_stop=True, post_think=False),
    "boxed-match-v1": lambda **kw: BoxedMatchScorer(scorer_id="boxed-match-v1",
                                                    require_stop=False, post_think=False),
    "post-think-v1": lambda **kw: BoxedMatchScorer(scorer_id="post-think-v1",
                                                   require_stop=True, post_think=True),
    "leak-filter-v1": lambda **kw: LeakFilterScorer(**kw),
}


def get_scorer(scorer_id: str = "boxed-match-stop-v1", **kw) -> Scorer:
    if scorer_id not in _REGISTRY:
        raise KeyError(f"unknown scorer {scorer_id!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[scorer_id](**kw)
