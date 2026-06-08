"""The vectorized per-position nucleus size must match the canonical recipe.

``_sequence_nucleus_stats`` computes sizes for a whole sequence at once (topk +
logsumexp), bypassing the per-position ``compute_nucleus`` sort. This checks the
two agree on random logits, position by position, plus the top-1 flag.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from math_rollouts.analysis.token_nuclei import _sequence_nucleus_stats
from math_rollouts.config import GenConfig
from math_rollouts.nucleus import compute_nucleus


def test_sizes_match_compute_nucleus():
    cfg = GenConfig()
    g = torch.Generator().manual_seed(7)
    T, V = 40, 3000
    logits = torch.randn(T, V, generator=g)
    chosen = torch.randint(0, V, (T,), generator=g)

    sizes, is_top1 = _sequence_nucleus_stats(
        logits, chosen, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)

    for t in range(T):
        ids, _ = compute_nucleus(logits[t], temperature=cfg.temperature,
                                 top_p=cfg.top_p, top_k=cfg.top_k)
        assert int(sizes[t]) == len(ids), t
        assert bool(is_top1[t]) == (int(chosen[t]) == ids[0]), t


def test_chunking_is_invariant():
    cfg = GenConfig()
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(50, 2000, generator=g)
    chosen = torch.randint(0, 2000, (50,), generator=g)
    a = _sequence_nucleus_stats(logits, chosen, temperature=cfg.temperature,
                                top_p=cfg.top_p, top_k=cfg.top_k, pos_chunk=1000)
    b = _sequence_nucleus_stats(logits, chosen, temperature=cfg.temperature,
                                top_p=cfg.top_p, top_k=cfg.top_k, pos_chunk=7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
