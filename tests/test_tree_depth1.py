"""Depth-1 parity: with ``max_depth=1`` the NucleusTree reproduces the legacy
``openings_k16`` first-token nucleus EXACTLY (token ids + renormalized probs), and
flattens to single-token openers whose ``branch_path == [i]`` and
``opener_token_ids == [fork_token_id]``.

CPU-only and deterministic: the nucleus recipe is checked against a fixed logits
vector (no model download); the end-to-end tree build uses a stub model returning
fixed logits. A real-model integration check on geometry/627 is left to the GPU run.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from math_rollouts.config import GenConfig
from math_rollouts.nucleus import NucleusTree, leaf_openers

T, TOP_P, TOPK = 0.6, 0.95, 20


def _legacy_nucleus(logits):
    """Verbatim openings_k16 first-token recipe (the parity target)."""
    probs = torch.softmax(logits.float() / T, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    keep = (torch.cumsum(sp, 0) - sp) < TOP_P
    keep[0] = True
    nuc_ids = si[keep][:TOPK]
    nuc_p = probs[nuc_ids]
    nuc_p = (nuc_p / nuc_p.sum()).tolist()
    nuc_ids = [int(t) for t in nuc_ids.tolist()]
    return nuc_ids, nuc_p


class _StubTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class _StubOut:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """Returns fixed last-position logits regardless of input — enough to drive the
    depth-1 nucleus extraction deterministically on CPU."""

    def __init__(self, logits_row):
        self._row = logits_row

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        b, seq = input_ids.shape
        v = self._row.shape[-1]
        out = torch.zeros((b, seq, v))
        out[0, -1] = self._row
        return _StubOut(out)


class _StubAdapter:
    model_id = "stub/model"
    is_thinking = False

    def terminal_ids(self, tok):
        return {}


def _fixed_logits(vocab=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, generator=g)


def test_nucleus_recipe_matches_legacy():
    cfg = GenConfig()
    logits = _fixed_logits()
    tree = NucleusTree(_StubModel(logits[None]), _StubTok(), _StubAdapter(), cfg,
                       max_depth=1, device="cpu")
    ids, probs = tree._nucleus(logits)
    lids, lprobs = _legacy_nucleus(logits)
    assert ids == lids
    assert len(probs) == len(lprobs)
    for a, b in zip(probs, lprobs):
        assert abs(a - b) < 1e-5


def test_depth1_build_produces_single_token_openers():
    cfg = GenConfig()
    logits = _fixed_logits()
    tok = _StubTok()
    tree = NucleusTree(_StubModel(logits[None]), tok, _StubAdapter(), cfg,
                       max_depth=1, device="cpu")
    root = tree.build([1, 2, 3])               # prompt ids irrelevant for the stub
    openers = leaf_openers(root, tok)
    lids, lprobs = _legacy_nucleus(logits)

    assert len(openers) == min(len(lids), cfg.top_k)
    for i, (op, tid) in enumerate(zip(openers, lids)):
        assert op["depth"] == 1
        assert op["branch_path"] == [i]
        assert op["opener_token_ids"] == [tid]
        assert op["fork_token_id"] == tid
        assert op["branch_size"] == len(lids)
        assert op["terminal"] is None
        # path_prob == inbound nucleus prob at depth 1.
        assert abs(op["path_prob"] - op["nuc_prob"]) < 1e-9
        assert abs(op["nuc_prob"] - lprobs[i]) < 1e-5
