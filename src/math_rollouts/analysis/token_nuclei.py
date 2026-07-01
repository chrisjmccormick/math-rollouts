"""Per-token nucleus store + size statistics over a rollout pool.

For every rollout in a naturally-sampled pool (e.g. ``math500_passK``) this
teacher-forces the completion through the model and, at each generated position,
records the **nucleus size** plus a compact, frugal slice of the distribution:

  * **singletons** (nucleus size 1): keep the top-2 tokens — enough to see whether
    the runner-up had any meaningful mass (a near-miss singleton), without the cost
    of a full distribution. Regenerate to investigate further.
  * **branches** (size >= 2): keep ``max(size, 10)`` tokens — at least 10, so the
    visualization can show alternates just *outside* the nucleus and a reachability
    check can tell "outside the nucleus but barely" from "far down". The nucleus
    size is recorded at its **true top-p extent** — uncapped, so on a flat
    distribution it can run to thousands of tokens — and the kept slice is uncapped
    too; in practice nuclei are tiny on average, so the store stays small. Because
    sizes are uncapped, ``nuc_sizes`` and ``keep_counts`` are stored as **int32**
    (int16 would overflow a >32767-token nucleus to a negative).

Stored per kept entry: the **raw logit** (pre-temperature — recompute any T/prob)
and the token **id**. Recompute precision is **bfloat16**, matching the vLLM engine
that generated the rollouts; that, plus the engine, is stamped in ``_meta.json`` so
the "just outside the nucleus" comparison is only ever made between like-precision
logits.

Output (a single ``token_nuclei.parquet`` — one per-rollout row per problem. The
earlier per-problem sharding produced up to 500 tiny files per pool that tripped
Windows path limits on clone/LFS checkout; nuclei stores stay small, so one file per
pool is ample. Named ``token_nuclei.parquet`` — not ``nuclei.parquet`` — to avoid
clashing with the experiment-level ``nuclei.parquet`` read by ``data.hf.load_nuclei``)::

    <out_dir>/generations/<model-slug>/<pool>_token_nuclei/
        token_nuclei.parquet  per-rollout rows; see _rows_to_table for the columns
        _stats.json           headline counts only (rollouts / tokens / singleton %);
                              full size, difficulty, and correct/incorrect breakdowns
                              are computed post-hoc from the store by
                              ``analysis.nuclei_stats.summarize_nuclei``
        _meta.json            model / engine / dtype / config / keep-rule

Per row the kept entries are stored FLAT (``kept_ids`` / ``kept_logits``) with a
parallel ``keep_counts`` so they can be re-split per position without a list-of-
lists column; see ``unpack_kept``. Rows are streamed one problem's row-group at a
time (``pyarrow.parquet.ParquetWriter``), so a per-``unique_id`` read stays cheap via
row-group statistics pushdown (see ``data.hf.load_token_nuclei``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import GenConfig
from ..data.hf import load_generation_parquet, load_problems_parquet, model_slug

# Keep-rule (also written to _meta.json). Branches keep >= BRANCH_MIN; there is no
# upper cap — the nucleus size is recorded uncapped, so a flat-distribution branch
# can keep (and report a size of) thousands of tokens.
SINGLETON_KEEP = 2
BRANCH_MIN = 10


def _sequence_kept(gen_logits, chosen_ids, *, temperature, top_p,
                   pos_chunk: int = 512):
    """Vectorized per-position TRUE top-p nucleus + frugal kept slice for one sequence.

    ``gen_logits`` [T, V] predict each completion token; ``chosen_ids`` [T] are
    those tokens. The nucleus size is the FULL top-p extent — uncapped, so on a flat
    distribution (where top-p "fails") it is recorded at its true width of hundreds
    or thousands of tokens rather than silently censored. Returns numpy arrays
    ``(sizes, is_top1, keep_counts, kept_ids_flat, kept_logits_flat)``; the flat
    arrays concatenate each position's kept entries (position-major, rank order) and
    split on ``keep_counts``. Sorting is chunked over positions (``pos_chunk``) to
    bound the [chunk, V] sort footprint regardless of rollout length.
    """
    import torch

    sizes_l, top1_l, keepn_l, ids_l, logit_l = [], [], [], [], []
    for s in range(0, gen_logits.shape[0], pos_chunk):
        gl = gen_logits[s:s + pos_chunk].float()
        ch = chosen_ids[s:s + pos_chunk]
        # Sort the FULL vocab by probability — identical to the canonical recipe
        # (``nucleus.recipe.compute_nucleus``) — so the store's nucleus matches the one
        # the viz/tree see, tail order included. The kept slice stores the RAW logit
        # (pre-temperature — recompute any T/prob downstream), gathered into that order.
        pr = torch.softmax(gl / temperature, dim=-1)
        srt_prob, srt_idx = torch.sort(pr, dim=-1, descending=True)
        srt_raw = torch.gather(gl, -1, srt_idx)
        csum = torch.cumsum(srt_prob, dim=-1)
        keep = (csum - srt_prob) < top_p
        keep[:, 0] = True
        size = keep.sum(dim=-1)                                   # [t], 1..V (uncapped)
        keep_n = torch.where(size <= 1, torch.full_like(size, SINGLETON_KEEP),
                             torch.clamp(size, min=BRANCH_MIN))
        kmax = int(keep_n.max())            # gather only the columns some position keeps
        store = torch.arange(kmax, device=gl.device).unsqueeze(0) < keep_n.unsqueeze(1)
        sizes_l.append(size.to(torch.int32))    # uncapped -> can exceed int16
        top1_l.append(srt_idx[:, 0] == ch)
        keepn_l.append(keep_n.to(torch.int32))  # = size for branches, so also uncapped
        ids_l.append(srt_idx[:, :kmax][store])
        logit_l.append(srt_raw[:, :kmax][store])
    return (torch.cat(sizes_l).cpu().numpy(), torch.cat(top1_l).cpu().numpy(),
            torch.cat(keepn_l).cpu().numpy(), torch.cat(ids_l).cpu().numpy(),
            torch.cat(logit_l).float().cpu().numpy())


def unpack_kept(row) -> list[tuple[list[int], list[float]]]:
    """Re-split a stored row's flat ``kept_ids``/``kept_logits`` into one
    ``(ids, logits)`` pair per generated position, using ``keep_counts``."""
    ids, logits, counts = row["kept_ids"], row["kept_logits"], row["keep_counts"]
    out, off = [], 0
    for c in counts:
        c = int(c)
        out.append((list(ids[off:off + c]), list(logits[off:off + c])))
        off += c
    return out


def _pack_batches(items, max_batch_tokens: int):
    """Greedily pack length-sorted items into padded batches under a token budget."""
    batches, cur, cur_max = [], [], 0
    for it in items:
        new_max = max(cur_max, it["seq_len"])
        if cur and (len(cur) + 1) * new_max > max_batch_tokens:
            batches.append(cur)
            cur, cur_max, new_max = [], 0, it["seq_len"]
        cur.append(it)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _rows_to_table(rows: list[dict], pa_logit):
    """Arrow table for a batch of per-rollout rows. Column types are explicit (never
    inferred from data), so every problem's row-group carries an identical schema —
    safe to stream through a single ``ParquetWriter`` even when a batch is all-null in
    some column, or empty."""
    import numpy as np
    import pyarrow as pa

    # PyArrow float16 rejects Python floats; cast numpy arrays to the target dtype first.
    np_logit = np.float16 if pa_logit == pa.float16() else np.float32
    col = lambda k, t: pa.array([r[k] for r in rows], type=t)
    return pa.table({
        "model_id": col("model_id", pa.string()),
        "unique_id": col("unique_id", pa.string()),
        "subject": col("subject", pa.string()),
        "sample_idx": col("sample_idx", pa.int16()),
        "run_id": col("run_id", pa.int32()),
        "answer_matches": col("answer_matches", pa.bool_()),
        "dup_index": col("dup_index", pa.int32()),
        "n_tokens": col("n_tokens", pa.int32()),
        "nuc_sizes": col("nuc_sizes", pa.list_(pa.int32())),
        "chosen_is_top1": col("chosen_is_top1", pa.list_(pa.bool_())),
        "keep_counts": col("keep_counts", pa.list_(pa.int32())),
        "kept_ids": col("kept_ids", pa.list_(pa.int32())),
        "kept_logits": pa.array([r["kept_logits"].astype(np_logit) for r in rows],
                                type=pa.list_(pa_logit)),
    })


def build_token_nuclei(model_id: str, pool: str, out_dir: str | Path, *,
                       temperature: float | None = None, top_p: float | None = None,
                       limit: int | None = None,
                       max_batch_tokens: int = 24000, device: str = "cuda",
                       logit_dtype: str = "float16", progress_every: int = 50):
    """Compute the per-token nucleus store for ``pool`` and write a single
    ``token_nuclei.parquet`` + ``_stats.json`` + ``_meta.json``. Returns
    ``(stats, paths)``. Nucleus sizes are recorded at their true top-p extent
    (uncapped). Rows are streamed one problem's row-group at a time, so peak memory
    stays bounded regardless of pool size, and a per-``unique_id`` read stays cheap.

    ``temperature`` / ``top_p`` set the regime at which the per-token nucleus is
    measured. Default to ``GenConfig()`` (T=0.6, top_p=0.95) -- the canonical
    thinking-pool regime. For self-consistent analysis on a non-canonical pool,
    pass the same regime the pool was sampled with."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..adapters import get_adapter

    cfg = GenConfig()
    if temperature is None:
        temperature = cfg.temperature
    if top_p is None:
        top_p = cfg.top_p
    adapter = get_adapter(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"Loading {model_id} on {device} (bfloat16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True).to(device).eval()
    pa_logit = {"float32": pa.float32(), "float16": pa.float16()}[logit_dtype]

    pool_df = load_generation_parquet(model_id, pool)
    if limit is not None:
        pool_df = pool_df.head(limit)
    prob_text = dict(zip(*[load_problems_parquet("math_problems")[c]
                           for c in ("unique_id", "problem")]))

    # Group rollouts by problem; cache the (repeated) prompt per problem.
    by_problem: dict[str, list] = {}
    prompt_cache: dict[str, list[int]] = {}
    missing = 0
    for r in pool_df.to_dict("records"):
        uid = r["unique_id"]
        if uid not in prob_text:
            missing += 1
            continue
        if uid not in prompt_cache:
            prompt_cache[uid] = adapter.prompt_ids({"problem": prob_text[uid]}, tok)
        comp = [int(t) for t in r["completion_token_ids"]]
        if comp:
            by_problem.setdefault(uid, []).append((r, comp))
    if missing:
        print(f"WARNING: {missing} rollouts skipped (unique_id not in problems table)",
              flush=True)
    problems = sorted(by_problem, key=lambda u: (by_problem[u][0][0].get("subject") or "", u))
    n_roll = sum(len(v) for v in by_problem.values())
    print(f"{len(problems)} problems, {n_roll} rollouts -> token_nuclei.parquet", flush=True)

    slug = model_slug(model_id)
    out_dir_path = Path(out_dir) / "generations" / slug / f"{pool}_token_nuclei"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    nuclei_path = out_dir_path / "token_nuclei.parquet"

    # Headline counters only. The full size distribution, per-difficulty bands, and
    # correct/incorrect splits are now computed POST-HOC from the written store by
    # ``math_rollouts.analysis.nuclei_stats.summarize_nuclei`` — keeping this compute
    # path free of analysis/difficulty concerns.
    n_tokens = singleton_count = n_done = problems_done = 0

    # Stream one problem's rows per row-group into a single file. Row-groups are
    # per-``unique_id``, so parquet column statistics let a filtered read skip
    # non-matching groups (see data.hf.load_token_nuclei). The writer is opened lazily
    # from the first row-group's schema so it matches byte-for-byte.
    writer = None
    for uid in problems:
        items = [{"row": r, "prompt": prompt_cache[uid], "comp": comp,
                  "seq_len": len(prompt_cache[uid]) + len(comp)}
                 for r, comp in by_problem[uid]]
        items.sort(key=lambda x: x["seq_len"])
        rows: list[dict] = []
        for batch in _pack_batches(items, max_batch_tokens):
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
                sizes, top1, keepn, ids_flat, logit_flat = _sequence_kept(
                    logits[i, plen - 1:plen - 1 + T], input_ids[i, plen:plen + T],
                    temperature=temperature, top_p=top_p)
                r = it["row"]
                rows.append({
                    "model_id": model_id, "unique_id": r["unique_id"],
                    "subject": r.get("subject"),
                    "sample_idx": int(r["sample_idx"]),
                    "run_id": int(r["run_id"]) if r.get("run_id") is not None else None,
                    "answer_matches": bool(r["answer_matches"]),
                    "dup_index": int(r["dup_index"]) if r.get("dup_index") is not None else None,
                    "n_tokens": int(T),
                    "nuc_sizes": sizes.astype("int32").tolist(),
                    "chosen_is_top1": top1.astype(bool).tolist(),
                    "keep_counts": keepn.astype("int32").tolist(),
                    "kept_ids": ids_flat.astype("int32").tolist(),
                    "kept_logits": logit_flat,
                })
                n_tokens += T
                singleton_count += int((sizes == 1).sum())
                n_done += 1
            del logits
        if rows:
            table = _rows_to_table(rows, pa_logit)
            if writer is None:
                writer = pq.ParquetWriter(nuclei_path, table.schema, compression="zstd")
            writer.write_table(table)
        problems_done += 1
        if progress_every and problems_done % progress_every == 0:
            print(f"  {problems_done}/{len(problems)} problems, {n_done}/{n_roll} rollouts",
                  flush=True)
    if writer is None:                          # pool had no scorable rollouts
        pq.write_table(_rows_to_table([], pa_logit), nuclei_path, compression="zstd")
    else:
        writer.close()

    stats = {
        "n_rollouts": n_done,
        "n_tokens": n_tokens,
        "singleton_count": singleton_count,
        "singleton_frac": float(singleton_count / n_tokens) if n_tokens else float("nan"),
    }
    meta = {
        "model_id": model_id, "pool": pool,
        "engine": "hf-teacher-forced", "dtype": "bfloat16",
        "logits": "raw (pre-temperature)", "logit_storage_dtype": logit_dtype,
        "temperature": float(temperature), "top_p": float(top_p),
        "nucleus_size": "true top-p extent (uncapped)",
        "keep_rule": {"singleton_keep": SINGLETON_KEEP, "branch_min": BRANCH_MIN,
                      "branch_max": "uncapped"},
        "n_rollouts": n_done, "n_problems": len(problems),
        "layout": "single token_nuclei.parquet; one row-group per unique_id",
        "columns": "kept_ids/kept_logits are FLAT, split per position on keep_counts "
                   "(see math_rollouts.analysis.token_nuclei.unpack_kept)",
    }
    (out_dir_path / "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir_path / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print_summary(stats, pool, model_id)
    print(f"\nwrote token_nuclei.parquet ({n_done} rollouts) + _stats.json + _meta.json "
          f"to {out_dir_path}", flush=True)
    return stats, {"dir": out_dir_path, "nuclei": nuclei_path,
                   "stats": out_dir_path / "_stats.json",
                   "meta": out_dir_path / "_meta.json"}


def _print_summary(stats: dict, pool: str, model_id: str) -> None:
    print(f"\n=== nucleus store: {model_id} / {pool} ===")
    print(f"  rollouts: {stats['n_rollouts']:,}   tokens: {stats['n_tokens']:,}")
    print(f"  SINGLETON nuclei: {stats['singleton_frac']*100:.1f}% "
          f"({stats['singleton_count']:,} / {stats['n_tokens']:,})")
    print("  full size / difficulty / correctness stats: compute post-hoc from the "
          "shards via\n  math_rollouts.analysis.nuclei_stats.summarize_nuclei "
          "(see notebook '03 - Analyze Rollout Nuclei').")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B", dest="model_id")
    ap.add_argument("--pool", default="math500_passK")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--logit-dtype", default="float16", choices=["float16", "float32"],
                    help="float16 (default) matches the bf16 compute precision and halves "
                         "the logit bytes; use float32 if your pyarrow can't write float16")
    ap.add_argument("--max-batch-tokens", type=int, default=24000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build_token_nuclei(a.model_id, a.pool, a.out_root, limit=a.limit,
                       logit_dtype=a.logit_dtype,
                       max_batch_tokens=a.max_batch_tokens, device=a.device)


if __name__ == "__main__":
    main()
