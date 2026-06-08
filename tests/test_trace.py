"""trace_nuclei parity + bookkeeping, CPU-only with a stub model (no download).

The stub returns a fixed last-position logits row for whatever single-token forward
it is given; here we drive a full-sequence forward and check that each step's nucleus
matches the shared recipe and that chosen_prob reflects nucleus membership.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from math_rollouts.config import GenConfig
from math_rollouts.nucleus import compute_nucleus, trace_nuclei


class _StubTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class _StubOut:
    def __init__(self, logits):
        self.logits = logits


class _PerPositionModel:
    """Returns distinct, deterministic logits at every sequence position, so each
    teacher-forced step has its own nucleus."""

    def __init__(self, vocab=4000):
        self.vocab = vocab

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        b, seq = input_ids.shape
        out = torch.zeros((b, seq, self.vocab))
        for pos in range(seq):
            g = torch.Generator().manual_seed(1000 + pos)
            out[0, pos] = torch.randn(self.vocab, generator=g)
        return _StubOut(out)


def test_trace_matches_recipe_per_step():
    cfg = GenConfig()
    model, tok = _PerPositionModel(), _StubTok()
    prompt_ids = [1, 2, 3]
    completion = [10, 20, 30, 40]

    steps = trace_nuclei(model, tok, prompt_ids, completion, cfg, device="cpu")
    assert len(steps) == len(completion)

    # Recompute the expected logits row that predicts each completion token and
    # confirm the step's nucleus matches the shared recipe exactly.
    full = model(torch.tensor([prompt_ids + completion])).logits[0]
    base = len(prompt_ids)
    for t, chosen in enumerate(completion):
        row = full[base + t - 1].float()
        ids, probs = compute_nucleus(row, temperature=cfg.temperature,
                                     top_p=cfg.top_p, top_k=cfg.top_k)
        assert steps[t]["step"] == t
        assert steps[t]["chosen_id"] == chosen
        assert steps[t]["chosen_str"] == f"<{chosen}>"
        assert steps[t]["nuc_ids"] == ids
        assert steps[t]["nuc_probs"] == probs
        # chosen_prob is the nucleus prob if the chosen token is a member, else 0.
        expected = dict(zip(ids, probs)).get(chosen, 0.0)
        assert steps[t]["chosen_prob"] == pytest.approx(expected)


def test_trace_requires_prompt():
    with pytest.raises(ValueError):
        trace_nuclei(_PerPositionModel(), _StubTok(), [], [10], GenConfig(), device="cpu")
