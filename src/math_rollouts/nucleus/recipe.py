"""The nucleus recipe — single source of truth for the top-p fan-out.

Both the branch tree (``tree.py``) and the teacher-forced rollout trace
(``trace.py``) compute their nucleus here, so there is exactly ONE definition of
"the nucleus" across exploration, visualization, and (future) reachability work.

Recipe (identical to the legacy ``openings_k16`` first-token rule): softmax on
temperature-scaled logits, keep the minimal top-p set (always keep the top
token), then renormalize within the kept set. The nucleus is the TRUE top-p
extent — uncapped, so a flat distribution's nucleus is reported at full width.
A caller that needs to bound fan-out (e.g. the exponential branch walk in
``tree.py``) limits how many members it *expands*, not what the nucleus *is*.
"""
from __future__ import annotations

import torch


def compute_nucleus(
    logits: torch.Tensor, *, temperature: float, top_p: float,
) -> tuple[list[int], list[float]]:
    """Return ``(ids, probs)`` parallel lists for the renormalized nucleus.

    ``ids`` are vocab ids in descending probability order; ``probs`` are the
    nucleus members' probabilities renormalized to sum to 1. Probabilities are
    rounded to 6 dp for stable, compact storage (matches the legacy recipe)."""
    pr = torch.softmax(logits.float() / temperature, dim=-1)
    sp, si = torch.sort(pr, descending=True)
    keep = (torch.cumsum(sp, 0) - sp) < top_p
    keep[0] = True
    ids = [int(t) for t in si[keep].tolist()]
    p = sp[keep]
    p = (p / p.sum()).tolist()
    return ids, [round(x, 6) for x in p]
