"""``extend_truncated`` (prefill+continue) — the faithful length-extend.

These tests stay model-free: they exercise the input filtering and the row-assembly
contract by feeding a fake LLM. The integration path (real vLLM) is left to manual
runs; see ``agent-ops/math-rollouts/.../fix_siphon.py`` for the one-off that
exercised the function on production data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from math_rollouts.config import GenConfig
from math_rollouts.generate.natural import extend_truncated


def _row(uid="u", sample_idx=0, terminal="truncated", ids=(1, 2, 3),
         text="prefix", **kw):
    """Minimal POOL_SCHEMA-shaped record sufficient for extend_truncated."""
    return {
        "model_id": "Qwen/Qwen3-8B", "unique_id": uid,
        "subject": "algebra", "answer": "42",
        "depth": 0, "branch_path": [], "opener_token_ids": [],
        "run_id": 0, "gen_config_id": 200, "seed": None,
        "temperature": 0.6, "top_p": 0.95, "max_gen_len": 8,
        "sample_idx": sample_idx,
        "completion_token_ids": list(ids), "completion_text": text,
        "finish_reason": "length", "stop_reason": None, "terminal": terminal,
        "prompt_num_tokens": 5, "completion_num_tokens": len(ids),
        "total_num_tokens": 5 + len(ids),
        **kw,
    }


def test_no_truncated_rows_returns_empty_no_model_load():
    """Fast-path: an all-EOS pool returns [] without importing vllm or loading a tok."""
    pool = pd.DataFrame([_row(terminal="emitted_eos"),
                         _row(terminal="stop_string")])
    # No llm/tok provided, no problems list — function must not need any of these
    # when the truncated subset is empty.
    out = extend_truncated("Qwen/Qwen3-8B", pool, [],
                           cfg=GenConfig(max_tokens=16), run_id=99)
    assert out == []


class _FakeOutput:
    def __init__(self, token_ids, text=" tail", finish_reason="stop", stop_reason=None):
        self.token_ids = list(token_ids)
        self.text = text
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason


class _FakeReq:
    def __init__(self, outputs):
        self.outputs = outputs


class _FakeLLM:
    """Captures (prompt_token_ids, max_tokens) per call, returns scripted outputs."""

    def __init__(self, scripted_continuations):
        self._cont = list(scripted_continuations)
        self.calls = []          # list of (prompt_ids, max_tokens, n)

    def generate(self, prompts, sps):
        outs = []
        for p, sp in zip(prompts, sps):
            self.calls.append((tuple(p["prompt_token_ids"]),
                               int(sp.max_tokens), int(sp.n)))
            outs.append(_FakeReq([self._cont.pop(0)]))
        return outs


class _FakeTok:
    eos_token_id = 999


def test_full_trajectory_is_prefix_plus_continuation(monkeypatch):
    """Returned row's completion_token_ids = original truncated ids + continuation;
    counts exclude trailing EOS; max_tokens for the continuation = budget - prefix."""
    pool = pd.DataFrame([_row(uid="u1", ids=(10, 11, 12), text="pre",
                              prompt_num_tokens=5)])
    problems = [{"unique_id": "u1", "problem": "p?", "answer": "42",
                 "subject": "algebra"}]

    fake_llm = _FakeLLM([_FakeOutput(token_ids=[20, 21, 999],
                                     text=" tail")])    # 999 is EOS in _FakeTok
    fake_tok = _FakeTok()

    # Stub the adapter so we don't need a tokenizer to build prompt_ids.
    class _Adapter:
        def prompt_ids(self, p, tok): return [100, 101, 102, 103, 104]
        def vllm_stop(self): return []
        def sampling_overrides(self): return {}
    monkeypatch.setattr("math_rollouts.adapters.get_adapter", lambda _mid: _Adapter())

    cfg = GenConfig(max_tokens=10, max_model_len=64)
    out = extend_truncated("Qwen/Qwen3-8B", pool, problems,
                           cfg=cfg, run_id=7, seed=None,
                           llm=fake_llm, tok=fake_tok)
    assert len(out) == 1
    r = out[0]
    # Full trajectory = prefix (3 ids) + continuation (3 ids incl. EOS) = 6 ids.
    assert r["completion_token_ids"] == [10, 11, 12, 20, 21, 999]
    # EOS-excluded token count.
    assert r["completion_num_tokens"] == 5
    # The vLLM call saw prompt+prefix as the prompt, and max_tokens = budget - prefix.
    (prompt_ids, mt, n), = fake_llm.calls
    assert prompt_ids == (100, 101, 102, 103, 104, 10, 11, 12)
    assert mt == 10 - 3                         # cfg.max_tokens - len(prefix)
    assert n == 1
    # Provenance: run_id from caller, sample_idx preserved, prompt_num_tokens is the
    # ORIGINAL problem prompt length (NOT incl. the prefilled prefix).
    assert r["run_id"] == 7 and r["sample_idx"] == 0
    assert r["prompt_num_tokens"] == 5
    assert r["max_gen_len"] == 10
    # Text concatenation (informational; the IDs are the source of truth).
    assert r["completion_text"] == "pre tail"


def test_rows_at_or_over_budget_are_skipped(monkeypatch):
    """A truncated row whose existing length already meets/exceeds the extend budget
    is silently skipped — there's nothing to add and SamplingParams would reject."""
    pool = pd.DataFrame([
        _row(uid="u1", ids=tuple(range(10)),   # 10 tokens; budget == 10
             prompt_num_tokens=5),
        _row(uid="u2", ids=tuple(range(12)),   # 12 tokens; budget < prefix
             prompt_num_tokens=5),
    ])
    problems = [{"unique_id": uid, "problem": "?", "answer": "42",
                 "subject": "x"} for uid in ("u1", "u2")]

    class _Adapter:
        def prompt_ids(self, p, tok): return [1, 2, 3]
        def vllm_stop(self): return []
        def sampling_overrides(self): return {}
    monkeypatch.setattr("math_rollouts.adapters.get_adapter", lambda _mid: _Adapter())

    fake_llm = _FakeLLM([])    # must not be called
    out = extend_truncated("any-model", pool, problems,
                           cfg=GenConfig(max_tokens=10, max_model_len=32),
                           run_id=1, llm=fake_llm, tok=_FakeTok())
    assert out == []
    assert fake_llm.calls == []


def test_rows_with_missing_problem_are_skipped(monkeypatch):
    """A truncated row whose unique_id isn't in `problems` is skipped (we can't
    rebuild the prompt without the problem)."""
    pool = pd.DataFrame([_row(uid="missing", ids=(1, 2)),
                         _row(uid="known", ids=(3, 4))])
    problems = [{"unique_id": "known", "problem": "?", "answer": "42",
                 "subject": "x"}]

    class _Adapter:
        def prompt_ids(self, p, tok): return [9, 9]
        def vllm_stop(self): return []
        def sampling_overrides(self): return {}
    monkeypatch.setattr("math_rollouts.adapters.get_adapter", lambda _mid: _Adapter())

    fake_llm = _FakeLLM([_FakeOutput(token_ids=[5])])
    out = extend_truncated("any-model", pool, problems,
                           cfg=GenConfig(max_tokens=20, max_model_len=64),
                           run_id=1, llm=fake_llm, tok=_FakeTok())
    assert len(out) == 1
    assert out[0]["unique_id"] == "known"
    assert len(fake_llm.calls) == 1
