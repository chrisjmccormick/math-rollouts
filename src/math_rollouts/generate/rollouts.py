"""The unified generator: nucleus pass (HF) then forced-rollout pass (vLLM).

Two phases, model-agnostic — every model-family difference is delegated to the
``ModelAdapter`` (prompt construction, terminals, stop strings):

  build_nuclei(...)   HF: for each problem, build the NucleusTree and flatten its
                      leaves into OPENER rows (NUCLEI_SCHEMA). One opener == one
                      forced prefix == one leaf of the tree.
  force_rollouts(...) vLLM: for each opener, force K samples through
                      ``prompt_ids + opener_token_ids`` and emit RAW rollout rows
                      (ROLLOUTS_SCHEMA) — NO correctness. Scoring is a separate pass.

``max_depth=1`` makes phase 1 the classic first-token nucleus (one single-token
opener per nucleus member) — byte-parity with the legacy ``openings_k16`` recipe.

The two phases are deliberately decoupled: phase 1 needs an HF model + KV cache;
phase 2 needs a vLLM engine. Run them in one process (free the HF model before
constructing the vLLM engine) or persist nuclei between processes.
"""
from __future__ import annotations

from ..config import GenConfig
from ..nucleus import NucleusTree, leaf_openers


def build_nuclei(model, tok, adapter, problems, cfg: GenConfig, *,
                 max_depth: int = 1, max_branch: int | None = None,
                 device: str = "cuda", progress_every: int = 100) -> list[dict]:
    """Phase 1 (HF). Build the nucleus tree per problem and return OPENER rows.

    Each row conforms to ``NUCLEI_SCHEMA``: the problem-identity fields
    (``model_id``, ``unique_id``, ``math500_native_id``, ``subject``, ``answer``,
    ``is_thinking``) plus the tree-derived opener fields from ``leaf_openers``."""
    tree = NucleusTree(model, tok, adapter, cfg, max_depth=max_depth,
                       max_branch=max_branch, device=device)
    rows: list[dict] = []
    for i, p in enumerate(problems):
        prompt = adapter.prompt_ids(p, tok)
        root = tree.build(prompt)
        native = p["unique_id"] if str(p["unique_id"]).startswith("test/") else None
        for op in leaf_openers(root, tok):
            rows.append({
                "model_id": adapter.model_id,
                "unique_id": p["unique_id"],
                "math500_native_id": native,
                "subject": p.get("subject", p["unique_id"].split("/")[1]
                                  if "/" in p["unique_id"] else ""),
                "answer": p["answer"],
                "is_thinking": bool(adapter.is_thinking),
                **op,
            })
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[build_nuclei] {i + 1}/{len(problems)} problems, "
                  f"{len(rows)} openers", flush=True)
    print(f"[build_nuclei] done: {len(problems)} problems -> {len(rows)} openers",
          flush=True)
    return rows


def _prompt_ids_by_uid(adapter, tok, problems) -> dict[str, list[int]]:
    return {p["unique_id"]: adapter.prompt_ids(p, tok) for p in problems}


def force_rollouts(llm, tok, adapter, nuclei_rows, problems, cfg: GenConfig, *,
                   k: int, run_id: int, seed: int | None = None) -> list[dict]:
    """Phase 2 (vLLM). Force ``k`` samples through every opener; emit RAW rollout
    rows (``ROLLOUTS_SCHEMA``) with NO correctness.

    ``nuclei_rows`` is the output of ``build_nuclei`` (or a reloaded nuclei.parquet);
    ``problems`` supplies the prompt prefix per ``unique_id``. The forced prompt is
    ``prompt_ids + opener_token_ids``; the stored ``completion_token_ids`` /
    ``completion_text`` re-prepend the forced opener so each rollout stands alone."""
    from vllm import SamplingParams

    prompt_by_uid = _prompt_ids_by_uid(adapter, tok, problems)
    sp_kwargs = dict(n=k, temperature=cfg.temperature, top_p=cfg.top_p,
                     max_tokens=cfg.max_tokens, stop=adapter.vllm_stop())
    if seed is not None:
        sp_kwargs["seed"] = seed
    sp = SamplingParams(**sp_kwargs)

    prompts, owners = [], []
    for n in nuclei_rows:
        prefix = prompt_by_uid[n["unique_id"]]
        opener = list(n["opener_token_ids"])
        prompts.append({"prompt_token_ids": prefix + opener})
        owners.append(n)
    print(f"[force_rollouts] generating {len(prompts)} openers x n={k} = "
          f"{len(prompts) * k} rollouts ...", flush=True)
    outs = llm.generate(prompts, sp)

    gcid = cfg.gen_config_id()
    rollouts: list[dict] = []
    for n, o in zip(owners, outs):
        opener_ids = list(n["opener_token_ids"])
        opener_str = "".join(n["opener_token_strs"])
        for j, c in enumerate(o.outputs):
            full_ids = opener_ids + list(c.token_ids)   # incl. the forced opener
            rollouts.append({
                "model_id": adapter.model_id,
                "unique_id": n["unique_id"],
                "math500_native_id": n["math500_native_id"],
                "subject": n["subject"],
                "answer": n["answer"],
                "depth": n["depth"],
                "branch_path": list(n["branch_path"]),
                "opener_token_ids": opener_ids,
                "run_id": run_id,
                "gen_config_id": gcid,
                "seed": seed,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_gen_len": cfg.max_tokens,
                "sample_idx": j,
                "completion_token_ids": full_ids,
                "completion_text": opener_str + c.text,
                "num_tokens": len(full_ids),
                "finish_reason": c.finish_reason,
            })
    print(f"[force_rollouts] done: {len(rollouts)} raw rollout rows", flush=True)
    return rollouts
