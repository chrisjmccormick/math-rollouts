"""Positional leak filter: a correct rollout is a genuine keeper only if its
VERIFIED answer appears past KEEP_FRAC of the response (default >70% through).

Copied verbatim (behaviour-preserving) from the source project's
``openers/lib/positional_filter.py``. The position is found by locating each
``\\boxed{}`` body and sympy-verifying the small ones against gold — the earliest
hit that verifies is where the correct answer is first asserted. Early WRONG boxes
and echoed empty ``\\boxed{}`` instructions are ignored automatically.
"""
from __future__ import annotations

import re
from functools import lru_cache

from math_verify import parse, verify

KEEP_FRAC = 0.70

_NORM = lambda s: re.sub(r"\s+", "", str(s))
_ANSWER_IS_RE = re.compile(r"(?:the answer is|=)\s*\$?\\?b?o?x?e?d?\{?\s*([^.$\n]{1,40})",
                           re.IGNORECASE)


@lru_cache(maxsize=8192)
def _gold(answer: str):
    return parse(f"\\boxed{{{answer}}}")


@lru_cache(maxsize=20000)
def _content_matches(content: str, answer: str) -> bool:
    if _NORM(content) == _NORM(answer):
        return True
    try:
        return bool(verify(_gold(answer), parse(content)))
    except Exception:
        return False


def _iter_boxed(text: str):
    """Yield (start_char, body) for each ``\\boxed{...}``, brace-matched."""
    i = 0
    key = "\\boxed{"
    while True:
        idx = text.find(key, i)
        if idx == -1:
            return
        depth, j = 1, idx + len(key)
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        yield idx, text[idx + len(key): j - 1]
        i = idx + len(key)


def verified_answer_char_pos(text: str, answer: str):
    """Earliest char offset where the CORRECT answer is asserted, or None."""
    answer = str(answer)
    best = None
    for start, body in _iter_boxed(text):
        if _content_matches(body, answer):
            best = start if best is None else min(best, start)
    if best is not None:
        return best
    na = _NORM(answer)
    for m in _ANSWER_IS_RE.finditer(text):
        if na and na in _NORM(m.group(1)):
            return m.start()
    return None


def answer_token_frac(text, token_ids, answer, tokenizer):
    """Position of the verified answer as a fraction of the response (in tokens)."""
    pos = verified_answer_char_pos(text, answer)
    if pos is None:
        return None
    n_before = len(tokenizer.encode(text[:pos], add_special_tokens=False))
    return n_before / max(len(token_ids), 1)


def classify(is_correct, frac, keep_frac=KEEP_FRAC):
    """Return one of: 'incorrect', 'unlocated', 'leak', 'keeper'."""
    if not is_correct:
        return "incorrect"
    if frac is None:
        return "unlocated"
    return "keeper" if frac > keep_frac else "leak"
