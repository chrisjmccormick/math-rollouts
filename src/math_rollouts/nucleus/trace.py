"""Per-token nucleus trace over an ALREADY-GENERATED rollout (teacher forcing).

Where ``tree.py`` *explores* branches by sampling forward, ``trace_nuclei`` takes a
known completion (e.g. a rollout pulled from the HF dataset) and recovers, at every
response position, the post-warper nucleus the model would have sampled from there —
plus the probability it assigned to the token the rollout actually took.

This is the primitive behind the token-chip visualization, and the same primitive
"reachability" analysis needs: teacher-force model B over model A's rollout and read
off, per token, whether A's choice was inside B's nucleus and at what probability.

ENGINE CAVEAT. The probabilities are recomputed with *this* HF model. If the rollout
was sampled by a different engine (vLLM in the dataset's forced/natural runs), tiny
logit differences can move a token across the top-p boundary, so a token the rollout
took may occasionally land just OUTSIDE the recomputed nucleus (reported as
``chosen_prob == 0.0``). For visualization this is cosmetic; for a reachability
membership decision, pin the recompute engine and treat membership as engine-relative.

The whole prompt+completion is forwarded ONCE (not autoregressively): teacher forcing
needs only a single pass, since every "next token" is already known.
"""
from __future__ import annotations

from typing import Any

import torch

from .recipe import compute_nucleus


def trace_nuclei(
    model, tok, prompt_ids: list[int], completion_token_ids: list[int], cfg, *,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Teacher-force ``completion_token_ids`` after ``prompt_ids`` and return one
    record per completion token::

        {step, chosen_id, chosen_str, chosen_prob, nuc_ids, nuc_probs}

    ``chosen_prob`` is the chosen token's renormalized nucleus probability, or 0.0
    if it fell outside the recomputed nucleus (see the engine caveat above).
    ``nuc_ids``/``nuc_probs`` are the full nucleus at that step (descending prob), so
    a caller can persist any subset it needs. One forward pass over the full sequence.
    """
    prompt_ids = list(prompt_ids)
    comp = list(completion_token_ids)
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty (need a prior position to "
                         "predict the first completion token)")

    input_ids = torch.tensor([prompt_ids + comp], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0]                       # [L, V]
    base = len(prompt_ids)                        # logits[base-1] predicts comp[0]

    steps: list[dict[str, Any]] = []
    for t, chosen_id in enumerate(comp):
        row = logits[base + t - 1].float()
        ids, probs = compute_nucleus(
            row, temperature=cfg.temperature, top_p=cfg.top_p,
        )
        prob_by_id = dict(zip(ids, probs))
        chosen_id = int(chosen_id)
        steps.append({
            "step": t,
            "chosen_id": chosen_id,
            "chosen_str": tok.decode([chosen_id]),
            "chosen_prob": float(prob_by_id.get(chosen_id, 0.0)),
            "nuc_ids": ids,
            "nuc_probs": probs,
        })
    return steps
