"""Per-token nucleus *sizes* over a whole rollout pool — the data behind the
"how small are these nuclei / how many are singletons?" statistics.

For every rollout in a naturally-sampled pool (e.g. ``math500_passK``) this
teacher-forces the completion through the model and records, at each generated
position, the **nucleus size** (how many tokens survive the top-k cap + top-p
keep, the set sampling could actually have drawn from) and whether the token the
rollout took was the model's top-1. It does NOT store the nucleus members by
default — only the per-position size — which is all the size/singleton
statistics need and keeps the output small. (Reachability work that needs the
members can add a members table later; see ``nucleus/trace.py``.)

The nucleus rule is the project's single source of truth (``nucleus.recipe``):
softmax(logits/T), keep top-k by prob, keep the minimal top-p set (always the top
token). Recompute precision is **bfloat16**, matching the vLLM generation engine —
some first-token logits are nearly tied, so fp32 would reshuffle the nucleus.

Output (under ``<out_dir>/generations/<model-slug>/``):
  <pool>_token_nuclei.parquet   one row per rollout; per-token lists
  <pool>_nuclei_stats.json      aggregate summary (the blog numbers)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import GenConfig
from ..data.hf import load_generation_parquet, load_problems_parquet, model_slug


def _sequence_nucleus_stats(gen_logits, chosen_ids, *, temperature, top_p, top_k,
                            pos_chunk: int = 1024):
    """Vectorized per-position nucleus size + top-1 flag for one sequence.

    ``gen_logits`` [T, V] are the logits predicting each completion token;
    ``chosen_ids`` [T] are those tokens. Returns ``(sizes, is_top1)`` numpy arrays.
    Chunked over positions to bound the float32 working set.
    """
    import torch

    sizes, top1s = [], []
    T = gen_logits.shape[0]
    for s in range(0, T, pos_chunk):
        gl = gen_logits[s:s + pos_chunk].float()
        ch = chosen_ids[s:s + pos_chunk]
        scaled = gl / temperature
        lse = torch.logsumexp(scaled, dim=-1, keepdim=True)
        top_logit, top_idx = torch.topk(scaled, top_k, dim=-1)       # [t, k] desc
        top_prob = torch.exp(top_logit - lse)                         # normalized
        csum = torch.cumsum(top_prob, dim=-1)
        keep = (csum - top_prob) < top_p
        keep[:, 0] = True
        sizes.append(keep.sum(dim=-1).to(torch.int16))
        top1s.append(top_idx[:, 0] == ch)
    return (torch.cat(sizes).cpu().numpy(), torch.cat(top1s).cpu().numpy())


def _pack_batches(items, max_batch_tokens: int):
    """Greedily pack length-sorted items into padded batches whose padded cost
    (n * max_len) stays under ``max_batch_tokens`` — keeps long rollouts in small
    batches and short ones in big batches."""
    batches, cur, cur_max = [], [], 0
    for it in items:
        L = it["seq_len"]
        new_max = max(cur_max, L)
        if cur and (len(cur) + 1) * new_max > max_batch_tokens:
            batches.append(cur)
            cur, cur_max, new_max = [], 0, L
        cur.append(it)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def iter_rollout_records(model_id: str, pool: str, *, limit: int | None = None,
                         max_batch_tokens: int = 24000, device: str = "cuda",
                         progress_every: int = 200):
    """Teacher-force every rollout in ``pool`` and yield one record per rollout:
    identity fields + ``nuc_sizes`` / ``chosen_is_top1`` per-completion-token lists."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..adapters import get_adapter

    cfg = GenConfig()
    adapter = get_adapter(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {model_id} on {device} (bfloat16) ...", flush=True)
    # torch_dtype (not dtype) for compatibility across transformers versions: it is
    # accepted by both old and new releases; the newest only warns.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    pool_df = load_generation_parquet(model_id, pool)
    if limit is not None:
        pool_df = pool_df.head(limit)
    prob_text = dict(zip(*[load_problems_parquet("math_problems")[c]
                           for c in ("unique_id", "problem")]))

    # Build the work list (cache prompt ids per problem; they repeat across samples).
    prompt_cache: dict[str, list[int]] = {}
    items, missing = [], 0
    for r in pool_df.to_dict("records"):
        uid = r["unique_id"]
        if uid not in prob_text:
            missing += 1
            continue
        if uid not in prompt_cache:
            prompt_cache[uid] = adapter.prompt_ids({"problem": prob_text[uid]}, tok)
        comp = [int(t) for t in r["completion_token_ids"]]
        if not comp:
            continue
        items.append({"row": r, "prompt": prompt_cache[uid], "comp": comp,
                      "seq_len": len(prompt_cache[uid]) + len(comp)})
    if missing:
        print(f"WARNING: {missing} rollouts skipped (unique_id not in problems table)",
              flush=True)
    items.sort(key=lambda x: x["seq_len"])
    batches = _pack_batches(items, max_batch_tokens)
    print(f"{len(items)} rollouts in {len(batches)} batches "
          f"(<= {max_batch_tokens} padded tokens each)", flush=True)

    done, next_mark = 0, progress_every
    for batch in batches:
        bmax = max(it["seq_len"] for it in batch)
        input_ids = torch.full((len(batch), bmax), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), bmax), dtype=torch.long)
        for i, it in enumerate(batch):
            seq = it["prompt"] + it["comp"]
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        for i, it in enumerate(batch):
            plen, T = len(it["prompt"]), len(it["comp"])
            gen_logits = logits[i, plen - 1:plen - 1 + T]
            chosen = input_ids[i, plen:plen + T]
            sizes, top1 = _sequence_nucleus_stats(
                gen_logits, chosen, temperature=cfg.temperature,
                top_p=cfg.top_p, top_k=cfg.top_k)
            r = it["row"]
            yield {
                "model_id": model_id,
                "unique_id": r["unique_id"],
                "math500_native_id": r.get("math500_native_id"),
                "subject": r.get("subject"),
                "level": int(r["level"]) if r.get("level") is not None else None,
                "sample_idx": int(r["sample_idx"]),
                "run_id": int(r["run_id"]) if r.get("run_id") is not None else None,
                "is_correct": bool(r["is_correct"]),
                "n_tokens": T,
                "nuc_sizes": sizes.astype("int16").tolist(),
                "chosen_is_top1": top1.astype(bool).tolist(),
            }
        del logits
        done += len(batch)
        if progress_every and done >= next_mark:
            print(f"  {done}/{len(items)} rollouts", flush=True)
            next_mark += progress_every


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-rollout records into the headline statistics."""
    import numpy as np

    sizes, first_sizes, top1, corr_sizes, incorr_sizes = [], [], [], [], []
    for r in records:
        s = r["nuc_sizes"]
        sizes.extend(s)
        if s:
            first_sizes.append(s[0])
        top1.extend(r["chosen_is_top1"])
        (corr_sizes if r["is_correct"] else incorr_sizes).extend(s)

    a = np.asarray(sizes)
    fa = np.asarray(first_sizes)
    vals, counts = np.unique(a, return_counts=True)
    frac1 = lambda x: float(np.mean(np.asarray(x) == 1)) if len(x) else float("nan")
    return {
        "n_rollouts": len(records),
        "n_tokens": int(a.size),
        "singleton_count": int((a == 1).sum()),
        "singleton_frac": float((a == 1).mean()),
        "mean_size": float(a.mean()),
        "median_size": float(np.median(a)),
        "p90_size": float(np.percentile(a, 90)),
        "max_size": int(a.max()),
        "chosen_is_top1_frac": float(np.mean(top1)),
        "size_histogram": {int(k): int(v) for k, v in zip(vals, counts)},
        "first_token_mean_size": float(fa.mean()),
        "first_token_singleton_frac": frac1(fa),
        "singleton_frac_correct": frac1(corr_sizes),
        "singleton_frac_incorrect": frac1(incorr_sizes),
    }


def _print_summary(stats: dict, pool: str, model_id: str) -> None:
    h = stats["size_histogram"]
    print(f"\n=== nucleus-size statistics: {model_id} / {pool} ===")
    print(f"  rollouts: {stats['n_rollouts']:,}   tokens: {stats['n_tokens']:,}")
    print(f"  SINGLETON nuclei: {stats['singleton_frac']*100:.1f}% "
          f"({stats['singleton_count']:,} / {stats['n_tokens']:,})")
    print(f"  mean size {stats['mean_size']:.3f}   median {stats['median_size']:.0f}   "
          f"p90 {stats['p90_size']:.0f}   max {stats['max_size']}")
    print(f"  chose top-1 token: {stats['chosen_is_top1_frac']*100:.1f}% of positions")
    print(f"  first response token: mean size {stats['first_token_mean_size']:.2f}, "
          f"singleton {stats['first_token_singleton_frac']*100:.1f}%")
    print(f"  singleton frac — correct {stats['singleton_frac_correct']*100:.1f}% | "
          f"incorrect {stats['singleton_frac_incorrect']*100:.1f}%")
    top = sorted(h.items())[:8]
    print("  size histogram:", "  ".join(f"{k}:{v/stats['n_tokens']*100:.1f}%" for k, v in top))


def build_token_nuclei(model_id: str, pool: str, out_dir: str | Path, *,
                       limit: int | None = None, max_batch_tokens: int = 24000,
                       device: str = "cuda", progress_every: int = 2000):
    """Compute per-token nucleus sizes for ``pool``, write the parquet + stats json
    under ``out_dir/generations/<slug>/``, and return ``(df, stats, paths)``."""
    import pandas as pd

    records = list(iter_rollout_records(
        model_id, pool, limit=limit, max_batch_tokens=max_batch_tokens,
        device=device, progress_every=progress_every))
    stats = summarize(records)
    _print_summary(stats, pool, model_id)

    base = Path(out_dir) / "generations" / model_slug(model_id)
    base.mkdir(parents=True, exist_ok=True)
    pq = base / f"{pool}_token_nuclei.parquet"
    sj = base / f"{pool}_nuclei_stats.json"
    df = pd.DataFrame(records)
    df.to_parquet(pq, index=False)
    sj.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {pq}\nwrote {sj}", flush=True)
    return df, stats, {"parquet": pq, "stats": sj}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B", dest="model_id")
    ap.add_argument("--pool", default="math500_passK",
                    help="standalone pool name under generations/<slug>/ (default: math500_passK)")
    ap.add_argument("--out-root", default=".",
                    help="local dataset root (output -> generations/<slug>/...)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N rollouts (a few thousand already "
                         "pins the singleton fraction tightly)")
    ap.add_argument("--max-batch-tokens", type=int, default=24000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build_token_nuclei(a.model_id, a.pool, a.out_root, limit=a.limit,
                       max_batch_tokens=a.max_batch_tokens, device=a.device)


if __name__ == "__main__":
    main()
