"""Post-hoc statistics over a per-token nucleus store (CPU-only, no torch).

``analysis.token_nuclei`` is the *compute* side: it teacher-forces rollouts on a
GPU and writes per-problem shards (one row per rollout, with a per-token
``nuc_sizes`` list). This module is the *analysis* side — pure functions over a
DataFrame of those shard rows, so reports can reshape the pool (e.g. an even-K
subsample) and recompute size statistics without re-running the GPU job.

A shard-row DataFrame has at least: ``unique_id``, ``answer_matches``, ``nuc_sizes``
(per-token list, sizes in ``1..top_k``) and ``chosen_is_top1`` (per-token bool
list). ``load_token_nuclei_pool`` in ``data.hf`` returns exactly this.
"""
from __future__ import annotations

DEFAULT_TOP_K = 20  # GenConfig.top_k — stored nucleus sizes never exceed this.


def _percentile_from_counts(counts, q):
    """q-th percentile of a value distribution given as a histogram of counts."""
    import numpy as np
    total = counts.sum()
    target = q / 100.0 * total
    cum = np.cumsum(counts)
    return int(np.searchsorted(cum, target))


def size_histogram(df, top_k: int = DEFAULT_TOP_K):
    """Counts of nucleus sizes ``0..top_k`` summed over every token of every row."""
    import numpy as np
    counts = np.zeros(top_k + 1, dtype=np.int64)
    for arr in df["nuc_sizes"]:
        a = np.asarray(arr, dtype=np.int64)
        counts += np.bincount(a, minlength=top_k + 1)[:top_k + 1]
    return counts


def even_k_sample(df, k: int, *, seed: int = 0, id_col: str = "unique_id"):
    """Even-K subsample: keep only groups (problems) with at least ``k`` rows, then
    sample exactly ``k`` from each. De-confounds a pass@K pool, where harder
    problems carry more rollouts, by giving every problem equal weight.

    Returns the balanced frame. Inspect ``df.groupby(id_col).size()`` first to pick
    ``k`` (and see how many problems would be dropped)."""
    sizes = df.groupby(id_col).size()
    keep_ids = sizes[sizes >= k].index
    sub = df[df[id_col].isin(keep_ids)]
    return sub.groupby(id_col, group_keys=False).sample(n=k, random_state=seed)


def size_by_position(df, max_pos: int | None = None):
    """Per-position nucleus profile across rollouts. Returns ``(mean_size,
    singleton_frac, count)`` numpy arrays indexed by token position (0 = first
    generated token). ``count[i]`` is how many rollouts reached position ``i``, so
    the deep tail is averaged over fewer, longer rollouts."""
    import numpy as np
    if max_pos is None:
        max_pos = max((len(a) for a in df["nuc_sizes"]), default=0)
    sum_size = np.zeros(max_pos, dtype=np.int64)
    sum_singleton = np.zeros(max_pos, dtype=np.int64)
    count = np.zeros(max_pos, dtype=np.int64)
    for arr in df["nuc_sizes"]:
        a = np.asarray(arr, dtype=np.int64)
        n = min(len(a), max_pos)
        sum_size[:n] += a[:n]
        sum_singleton[:n] += (a[:n] == 1)
        count[:n] += 1
    denom = np.maximum(count, 1)
    return sum_size / denom, sum_singleton / denom, count


def _band_stats(counts, n_top1, ramp):
    """Headline numbers for one size histogram + its top-1 count."""
    n_tokens = int(counts.sum())
    if not n_tokens:
        return None
    return {
        "n_tokens": n_tokens,
        "singleton_count": int(counts[1]),
        "singleton_frac": float(counts[1] / n_tokens),
        "mean_size": float((ramp * counts).sum() / n_tokens),
        "chosen_is_top1_frac": float(n_top1 / n_tokens),
    }


def summarize_nuclei(df, *, top_k: int = DEFAULT_TOP_K, band_map: dict | None = None) -> dict:
    """Aggregate size statistics over a shard-row DataFrame.

    Reproduces the dict ``build_token_nuclei`` used to compute in its loop, but
    post-hoc from stored shards so it can run on any (e.g. subsampled) slice.

    ``band_map`` maps ``unique_id -> band`` (e.g. from
    ``analysis.difficulty.attach_bands``/``_band_map``); when given, a per-band
    breakdown is added under ``by_band``.
    """
    import numpy as np

    ramp = np.arange(top_k + 1)
    total = np.zeros(top_k + 1, dtype=np.int64)
    first = np.zeros(top_k + 1, dtype=np.int64)
    corr = np.zeros(top_k + 1, dtype=np.int64)
    incorr = np.zeros(top_k + 1, dtype=np.int64)
    n_top1 = n_top1_corr = n_top1_incorr = 0
    n_rollouts = 0

    bands = sorted(set(band_map.values())) if band_map else []
    band_counts = {b: np.zeros(top_k + 1, dtype=np.int64) for b in bands}
    band_top1 = {b: 0 for b in bands}

    for row in df.itertuples(index=False):
        a = np.asarray(row.nuc_sizes, dtype=np.int64)
        if a.size == 0:
            continue
        bc = np.bincount(a, minlength=top_k + 1)[:top_k + 1]
        t1 = int(np.asarray(row.chosen_is_top1, dtype=bool).sum())
        total += bc
        first[a[0]] += 1
        n_top1 += t1
        n_rollouts += 1
        if row.answer_matches:
            corr += bc
            n_top1_corr += t1
        else:
            incorr += bc
            n_top1_incorr += t1
        if band_map is not None:
            b = band_map.get(row.unique_id, "Unknown")
            if b not in band_counts:
                band_counts[b] = np.zeros(top_k + 1, dtype=np.int64)
                band_top1[b] = 0
            band_counts[b] += bc
            band_top1[b] += t1

    n_tokens = int(total.sum())
    frac1 = lambda c: float(c[1] / c.sum()) if c.sum() else float("nan")
    stats = {
        "n_rollouts": n_rollouts,
        "n_tokens": n_tokens,
        "singleton_count": int(total[1]),
        "singleton_frac": float(total[1] / n_tokens) if n_tokens else float("nan"),
        "mean_size": float((ramp * total).sum() / n_tokens) if n_tokens else float("nan"),
        "median_size": _percentile_from_counts(total, 50),
        "p90_size": _percentile_from_counts(total, 90),
        "chosen_is_top1_frac": float(n_top1 / n_tokens) if n_tokens else float("nan"),
        "size_histogram": {int(k): int(v) for k, v in enumerate(total) if v},
        "first_token_mean_size": float((ramp * first).sum() / first.sum()) if first.sum() else float("nan"),
        "first_token_singleton_frac": frac1(first),
        "singleton_frac_correct": frac1(corr),
        "singleton_frac_incorrect": frac1(incorr),
    }
    if band_map is not None:
        stats["by_band"] = {
            b: s for b in band_counts
            if (s := _band_stats(band_counts[b], band_top1[b], ramp)) is not None
        }
    return stats
