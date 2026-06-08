"""_sequence_kept: nucleus sizes match the canonical recipe, the keep-rule holds
(2 for singletons, >=10 capped at top_k for branches), and the kept set always
contains the full nucleus (so reachability can see "just outside").
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from math_rollouts.analysis.token_nuclei import (
    BRANCH_MIN, SINGLETON_KEEP, _sequence_kept, unpack_kept,
)
from math_rollouts.config import GenConfig
from math_rollouts.nucleus import compute_nucleus


def test_sizes_keeprule_and_nucleus_subset():
    cfg = GenConfig()
    g = torch.Generator().manual_seed(7)
    T, V = 60, 4000
    logits = torch.randn(T, V, generator=g)
    chosen = torch.randint(0, V, (T,), generator=g)

    sizes, is_top1, keepn, ids_flat, logit_flat = _sequence_kept(
        logits, chosen, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)

    # Flat arrays split on keep_counts and total to their sum.
    assert ids_flat.shape == logit_flat.shape == (int(keepn.sum()),)
    off = 0
    for t in range(T):
        ids, _ = compute_nucleus(logits[t], temperature=cfg.temperature,
                                 top_p=cfg.top_p, top_k=cfg.top_k)
        size = len(ids)
        assert int(sizes[t]) == size, t
        # keep-rule
        expect_keep = SINGLETON_KEEP if size == 1 else min(max(size, BRANCH_MIN), cfg.top_k)
        assert int(keepn[t]) == expect_keep, t
        # the kept set's first `size` ids ARE the nucleus (kept is a superset)
        kept_ids = ids_flat[off:off + expect_keep]
        assert list(kept_ids[:size]) == ids, t
        # stored logits are RAW (pre-temperature) logits of those ids
        assert np.allclose(logit_flat[off:off + size],
                           logits[t, kept_ids[:size]].numpy(), atol=1e-4), t
        off += expect_keep
        assert bool(is_top1[t]) == (int(chosen[t]) == ids[0])


def test_unpack_roundtrip():
    cfg = GenConfig()
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(20, 2000, generator=g)
    chosen = torch.randint(0, 2000, (20,), generator=g)
    _, _, keepn, ids_flat, logit_flat = _sequence_kept(
        logits, chosen, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
    row = {"keep_counts": keepn.tolist(), "kept_ids": ids_flat.tolist(),
           "kept_logits": logit_flat.tolist()}
    per_pos = unpack_kept(row)
    assert len(per_pos) == 20
    assert [len(ids) for ids, _ in per_pos] == keepn.tolist()
    assert sum(len(ids) for ids, _ in per_pos) == len(ids_flat)
