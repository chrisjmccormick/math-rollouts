"""Natural (unguided) sampling — the generator for the self-contained rollout pools.

Unlike ``generate.run`` (which forces openers from the nucleus tree), this samples
the completion from the bare prompt: the model picks its own first token, so the
first-token nucleus diversity is preserved. This is how the ``math500_passK`` /
``math12k_*`` pools are produced and extended.

One vLLM pass, no HF model, no nucleus tree. Emits ``ROLLOUTS_SCHEMA`` rows with
``depth=0, branch_path=[], opener_token_ids=[]`` (no forced opener). Scoring +
flat-pool assembly happen on CPU in ``data.pools``.
"""
from __future__ import annotations

from ..config import GenConfig


def generate_natural(model_id: str, problems: list[dict], *, k,
                     run_id: int, seed: int | None = None,
                     cfg: GenConfig | None = None, device: str = "cuda",
                     llm=None, tok=None) -> list[dict]:
    """Natural-sample completions for ``problems`` and return RAW rollout rows
    (``ROLLOUTS_SCHEMA``; NO correctness — score with ``data.pools.score_rollouts``).

    ``k`` is either an int (same number of samples for every problem) or a
    ``dict[unique_id, int]`` of per-problem sample counts — the latter drives a
    width-extend with exact per-problem deficits (problems mapping to ``0``/missing
    are skipped). ``run_id``/``seed`` mark batch identity; use a fresh ``run_id`` when
    extending so ``(unique_id, run_id, sample_idx)`` stays unique against prior rows.

    Pass an existing ``llm``/``tok`` to reuse one engine across calls; otherwise a
    vLLM engine is built for ``model_id``.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from ..adapters import get_adapter

    cfg = cfg or GenConfig()
    adapter = get_adapter(model_id)
    tok = tok or AutoTokenizer.from_pretrained(model_id)
    llm = llm or LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.9,
                     max_model_len=cfg.max_model_len)

    def _n_for(uid: str) -> int:
        return int(k) if not isinstance(k, dict) else int(k.get(uid, 0))

    def _sp(n: int):
        # Sampling is temperature + top-p ONLY. cfg.top_k is a post-hoc nucleus-size
        # cap (see analysis.token_nuclei); it is deliberately NOT a generation limiter.
        kw = dict(n=n, temperature=cfg.temperature, top_p=cfg.top_p,
                  max_tokens=cfg.max_tokens, stop=adapter.vllm_stop())
        if seed is not None:
            kw["seed"] = seed
        return SamplingParams(**kw)

    prompts, sps, owners = [], [], []
    for p in problems:
        n = _n_for(p["unique_id"])
        if n <= 0:
            continue
        prompts.append({"prompt_token_ids": adapter.prompt_ids(p, tok)})
        sps.append(_sp(n))
        owners.append(p)
    total = sum(sp.n for sp in sps)
    print(f"[generate_natural] {len(prompts)} problems -> {total} rollouts "
          f"(run_id={run_id}, seed={seed}) ...", flush=True)
    if not prompts:
        return []
    # vLLM accepts a per-prompt list of SamplingParams (variable n per problem).
    outs = llm.generate(prompts, sps if isinstance(k, dict) else sps[0])

    gcid = cfg.gen_config_id()
    rollouts: list[dict] = []
    for p, o in zip(owners, outs):
        for j, c in enumerate(o.outputs):
            ids = list(c.token_ids)
            rollouts.append({
                "model_id": model_id,
                "unique_id": p["unique_id"],
                "subject": p.get("subject"),
                "answer": p.get("answer"),
                "depth": 0,
                "branch_path": [],
                "opener_token_ids": [],
                "run_id": run_id,
                "gen_config_id": gcid,
                "seed": seed,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_gen_len": cfg.max_tokens,
                "sample_idx": j,
                "completion_token_ids": ids,
                "completion_text": c.text,
                "num_tokens": len(ids),
                "finish_reason": c.finish_reason,
            })
    print(f"[generate_natural] done: {len(rollouts)} raw rollout rows", flush=True)
    return rollouts
