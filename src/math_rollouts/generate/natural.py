"""Natural (unguided) sampling — the generator for the self-contained rollout pools.

Unlike ``generate.run`` (which forces openers from the nucleus tree), this samples
the completion from the bare prompt: the model picks its own first token, so the
first-token nucleus diversity is preserved. This is how the ``math500_passK`` /
``math12k_*`` pools are produced and extended.

Two entry points:
  generate_natural   one vLLM pass, K samples per problem -> ROLLOUTS_SCHEMA rows.
  extend_truncated   prefill+continue length-extend for the truncated subset of an
                     existing pool. Feeds prompt + already-generated tokens back as
                     the new vLLM prompt and samples a continuation up to the higher
                     budget. Faithful: by autoregressive factorization the extended
                     trajectory is statistically identical to one generated at the
                     larger budget from the start -- so it does NOT bias the pool
                     against long traces the way ``drop + redraw`` does.

No HF model, no nucleus tree. Emit ``ROLLOUTS_SCHEMA`` rows with
``depth=0, branch_path=[], opener_token_ids=[]`` (no forced opener). Scoring +
flat-pool assembly happen on CPU in ``data.pools``.
"""
from __future__ import annotations

from ..config import GenConfig, filenoable_stdio


def generate_natural(model_id: str, problems: list[dict], *, k,
                     run_id: int, seed: int | None = None,
                     cfg: GenConfig | None = None, device: str = "cuda",
                     llm=None, tok=None) -> list[dict]:
    """Natural-sample completions for ``problems`` and return RAW rollout rows
    (``ROLLOUTS_SCHEMA``; NO answer/match facts — those are added when the pool is
    built via ``data.pools.build_pool`` / ``compute_raw_attributes``).

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
    if llm is None:
        # vLLM V1 treats ENGINE seed None as 0 (a fixed RNG): an identical request
        # set re-run in a fresh session would regenerate identical completions. For
        # unseeded (natural) sampling the engine must get fresh entropy; when `seed`
        # IS set, per-request SamplingParams seeds govern and this just matches.
        import secrets
        engine_seed = seed if seed is not None else secrets.randbits(31)
        print(f"[generate_natural] engine seed {engine_seed}"
              f"{' (fresh entropy; rows stay unseeded)' if seed is None else ''}",
              flush=True)
        with filenoable_stdio():
            llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.9,
                      max_model_len=cfg.max_model_len, seed=engine_seed)

    def _n_for(uid: str) -> int:
        return int(k) if not isinstance(k, dict) else int(k.get(uid, 0))

    def _sp(n: int):
        # Sampling is temperature + top-p, PLUS the adapter's per-family overrides
        # (e.g. Qwen3's vendor thinking-mode top_k=20 — a real vLLM sampling limiter
        # owned by the adapter). Nucleus SIZE is measured post-hoc and uncapped (see
        # analysis.token_nuclei); there is no nucleus-size cap knob anymore.
        kw = dict(n=n, temperature=cfg.temperature, top_p=cfg.top_p,
                  max_tokens=cfg.max_tokens, stop=adapter.vllm_stop())
        kw.update(adapter.sampling_overrides())
        if seed is not None:
            kw["seed"] = seed
        return SamplingParams(**kw)

    eos_id = tok.eos_token_id
    prompts, sps, owners, prompt_lens = [], [], [], []
    for p in problems:
        n = _n_for(p["unique_id"])
        if n <= 0:
            continue
        pids = adapter.prompt_ids(p, tok)
        prompts.append({"prompt_token_ids": pids})
        sps.append(_sp(n))
        owners.append(p)
        prompt_lens.append(len(pids))
    total = sum(sp.n for sp in sps)
    print(f"[generate_natural] {len(prompts)} problems -> {total} rollouts "
          f"(run_id={run_id}, seed={seed}) ...", flush=True)
    if not prompts:
        return []
    # vLLM accepts a per-prompt list of SamplingParams (variable n per problem).
    outs = llm.generate(prompts, sps if isinstance(k, dict) else sps[0])

    from ..score.scorers import derive_terminal

    gcid = cfg.gen_config_id()
    rollouts: list[dict] = []
    for p, plen, o in zip(owners, prompt_lens, outs):
        for j, c in enumerate(o.outputs):
            ids = list(c.token_ids)
            stop_reason = None if c.stop_reason is None else str(c.stop_reason)
            # vLLM returns the trailing EOS as the final id; exclude it from the count.
            comp_n = len(ids) - 1 if (ids and ids[-1] == eos_id) else len(ids)
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
                "finish_reason": c.finish_reason,
                "stop_reason": stop_reason,
                "terminal": derive_terminal(c.finish_reason, stop_reason),
                "prompt_num_tokens": plen,
                "completion_num_tokens": comp_n,
                "total_num_tokens": plen + comp_n,
            })
    print(f"[generate_natural] done: {len(rollouts)} raw rollout rows", flush=True)
    return rollouts


def extend_truncated(model_id: str, pool_df, problems: list[dict], *,
                     cfg: GenConfig, run_id: int, seed: int | None = None,
                     llm=None, tok=None) -> list[dict]:
    """Prefill-and-continue extension for truncated rollouts: take each truncated row,
    feed its existing ``completion_token_ids`` back as part of the vLLM prompt, and
    sample a continuation up to ``cfg.max_tokens`` TOTAL completion tokens. The
    returned row's completion is the original truncated prefix CONCATENATED with the
    fresh continuation -- statistically identical to what the model would have
    produced if it had been given ``cfg.max_tokens`` budget from the start (by
    autoregressive factorization ``p(x|prompt) = p(prefix|prompt) * p(tail|prefix)``).

    Unlike the broken "drop + redraw" pattern, this does NOT bias the pool against
    long traces -- a trace that ran past the lower budget keeps its prefix and gets
    extended, instead of being discarded in favor of a fresh (and shorter-leaning)
    draw.

    ``pool_df`` is a POOL_SCHEMA frame (or anything carrying ``unique_id``,
    ``run_id``, ``sample_idx``, ``completion_token_ids``, ``terminal``,
    ``prompt_num_tokens``). Only rows with ``terminal == "truncated"`` are touched.
    ``cfg.max_tokens`` is the TOTAL completion budget at the extend ceiling (not the
    additional budget). ``cfg.max_model_len`` must be large enough for
    ``prompt + cfg.max_tokens``. ``problems`` supplies ``prompt_ids`` per ``unique_id``;
    only the problems referenced by truncated rows are used.

    Returns RAW rollout rows (``ROLLOUTS_SCHEMA``) -- one per extended trajectory,
    with the original ``sample_idx`` preserved (so callers can use a fresh
    ``run_id`` and the ``(uid, run_id, sample_idx)`` key stays unique against the
    pool). The caller swaps these into the pool (drop the truncated rows; append the
    new rows; rebuild attributes via ``data.pools.build_pool``)."""
    # Empty-case fast path: no truncated rows -> no model load, no vLLM import.
    trunc = pool_df[pool_df["terminal"] == "truncated"]
    if not len(trunc):
        return []

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from ..adapters import get_adapter

    adapter = get_adapter(model_id)
    tok = tok or AutoTokenizer.from_pretrained(model_id)
    prob_by_uid = {p["unique_id"]: p for p in problems}

    # Build (prompt_token_ids, per-row SamplingParams) for every truncated row.
    # The "prompt" we hand to vLLM is original-prompt + already-generated-tokens;
    # the SamplingParams.max_tokens caps the CONTINUATION (cfg.max_tokens - prefix).
    eos_id = tok.eos_token_id
    prompts: list[dict] = []
    sps: list[SamplingParams] = []
    src_rows: list[dict] = []
    prompt_lens: list[int] = []     # original prompt length (not incl. prefix)
    skipped = 0
    for r in trunc.to_dict("records"):
        uid = r["unique_id"]
        if uid not in prob_by_uid:
            skipped += 1
            continue
        prompt_ids = adapter.prompt_ids(prob_by_uid[uid], tok)
        prefix = [int(t) for t in r["completion_token_ids"]]
        remaining = int(cfg.max_tokens) - len(prefix)
        if remaining <= 0:
            # Already at/over the extend ceiling -- nothing to add.
            skipped += 1
            continue
        kw = dict(n=1, temperature=cfg.temperature, top_p=cfg.top_p,
                  max_tokens=remaining, stop=adapter.vllm_stop())
        kw.update(adapter.sampling_overrides())
        if seed is not None:
            kw["seed"] = seed
        prompts.append({"prompt_token_ids": prompt_ids + prefix})
        sps.append(SamplingParams(**kw))
        src_rows.append(r)
        prompt_lens.append(len(prompt_ids))
    print(f"[extend_truncated] {len(prompts)} truncated rows to extend "
          f"(skipped {skipped}; run_id={run_id}, seed={seed}, "
          f"budget={cfg.max_tokens}) ...", flush=True)
    if not prompts:
        return []

    if llm is None:
        import secrets
        engine_seed = seed if seed is not None else secrets.randbits(31)
        print(f"[extend_truncated] engine seed {engine_seed}"
              f"{' (fresh entropy; rows stay unseeded)' if seed is None else ''}",
              flush=True)
        with filenoable_stdio():
            llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.9,
                      max_model_len=cfg.max_model_len, seed=engine_seed)

    outs = llm.generate(prompts, sps)

    from ..score.scorers import derive_terminal

    gcid = cfg.gen_config_id()
    rollouts: list[dict] = []
    for r, plen, o in zip(src_rows, prompt_lens, outs):
        c = o.outputs[0]                                    # n=1
        prefix_ids = [int(t) for t in r["completion_token_ids"]]
        cont_ids = list(c.token_ids)
        full_ids = prefix_ids + cont_ids                    # complete trajectory
        stop_reason = None if c.stop_reason is None else str(c.stop_reason)
        comp_n = len(full_ids) - 1 if (full_ids and full_ids[-1] == eos_id) else len(full_ids)
        # The extended trajectory's text is the original truncated text + the
        # continuation text; vLLM only returns the continuation, so prepend the
        # original. (completion_text is for human inspection; the IDs are the truth.)
        full_text = (r.get("completion_text") or "") + (c.text or "")
        rollouts.append({
            "model_id": model_id,
            "unique_id": r["unique_id"],
            "subject": r.get("subject"),
            "answer": r.get("answer"),
            "depth": int(r.get("depth", 0) or 0),
            "branch_path": list(r.get("branch_path") or []),
            "opener_token_ids": list(r.get("opener_token_ids") or []),
            "run_id": run_id,
            "gen_config_id": gcid,
            "seed": seed,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_gen_len": cfg.max_tokens,
            "sample_idx": int(r["sample_idx"]),
            "completion_token_ids": full_ids,
            "completion_text": full_text,
            "finish_reason": c.finish_reason,
            "stop_reason": stop_reason,
            "terminal": derive_terminal(c.finish_reason, stop_reason),
            "prompt_num_tokens": int(plen),
            "completion_num_tokens": int(comp_n),
            "total_num_tokens": int(plen + comp_n),
        })
    print(f"[extend_truncated] done: {len(rollouts)} extended rollout rows", flush=True)
    return rollouts
