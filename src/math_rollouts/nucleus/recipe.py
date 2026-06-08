"""The nucleus recipe — single source of truth for the top-p/top-k fan-out.

Both the branch tree (``tree.py``) and the teacher-forced rollout trace
(``trace.py``) compute their nucleus here, so there is exactly ONE definition of
"the nucleus" across exploration, visualization, and (future) reachability work.

Recipe (identical to the legacy ``openings_k16`` first-token rule): softmax on
temperature-scaled logits, cap at ``top_k`` by probability, keep the minimal
top-p set (always keep the top token), then renormalize within the kept set.
"""
from __future__ import annotations

import torch


def compute_nucleus(
    logits: torch.Tensor, *, temperature: float, top_p: float, top_k: int,
) -> tuple[list[int], list[float]]:
    """Return ``(ids, probs)`` parallel lists for the renormalized nucleus.

    ``ids`` are vocab ids in descending probability order; ``probs`` are the
    nucleus members' probabilities renormalized to sum to 1. Probabilities are
    rounded to 6 dp for stable, compact storage (matches the legacy recipe)."""
    pr = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = torch.sort(pr, descending=True)
    sp, si = sp[:top_k], si[:top_k]
    keep = (torch.cumsum(sp, 0) - sp) < top_p
    keep[0] = True
    ids = [int(t) for t in si[keep].tolist()]
    p = sp[keep]
    p = (p / p.sum()).tolist()
    return ids, [round(x, 6) for x in p]
