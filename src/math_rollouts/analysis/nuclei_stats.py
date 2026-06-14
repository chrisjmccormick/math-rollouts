"""Post-hoc statistics over a per-token nucleus store (CPU-only, no torch).

``analysis.token_nuclei`` is the *compute* side: it teacher-forces rollouts on a
GPU and writes per-problem shards (one row per rollout, with a per-token
``nuc_sizes`` list). This module is the *analysis* side — pure functions over a
DataFrame of those shard rows, so reports can reshape the pool (e.g. an even-K
subsample) and recompute size statistics without re-running the GPU job.

A shard-row DataFrame has at least: ``unique_id``, ``answer_matches``, ``nuc_sizes``
(per-token list of true top-p sizes) and ``chosen_is_top1`` (per-token bool list).
``load_token_nuclei_pool`` in ``data.hf`` returns exactly this.

The compute side (``token_nuclei``) records the **true top-p size**, uncapped — which on
flat distributions can run to thousands of tokens. The linear ``size_histogram`` /
``summarize_nuclei`` aggregators bin sizes into ``[1, hist_max]`` for a readable head,
**folding** anything wider into the top bucket. That fold is a DISPLAY/binning bound, not
a nucleus or sampling cap: ``mean_size`` is computed from the true (unfolded) sizes, and
the full heavy tail is available via ``size_distribution``.
"""
from __future__ import annotations

SIZE_HIST_CAP = 20  # linear-histogram top bin; sizes >= this fold in for display only.


def _fold_sizes(a, hist_max):
    """Fold nucleus sizes into ``[1, hist_max]`` for the linear histogram.

    Sizes are stored uncapped as int32. Anything above ``hist_max`` is folded into the
    top bin (a display bound, NOT a nucleus cap); a value below 1 can only come from a
    stale int16-overflowed store and is likewise folded — keeping every value a valid
    bincount index in range."""
    import numpy as np
    return np.where((a < 1) | (a > hist_max), hist_max, a)


def _percentile_from_counts(counts, q):
    """q-th percentile of a value distribution given as a histogram of counts."""
    import numpy as np
    total = counts.sum()
    target = q / 100.0 * total
    cum = np.cumsum(counts)
    return int(np.searchsorted(cum, target))


def size_histogram(df, hist_max: int = SIZE_HIST_CAP):
    """Counts of nucleus sizes ``0..hist_max`` summed over every token of every row.

    Linear, single-token resolution — good for the small-size head (and the headline
    singleton bar), but it folds everything ``>= hist_max`` into the top bin. For the
    full heavy-tailed picture (sizes run uncapped to tens of thousands on flat
    distributions) use ``size_distribution``."""
    import numpy as np
    counts = np.zeros(hist_max + 1, dtype=np.int64)
    for arr in df["nuc_sizes"]:
        a = _fold_sizes(np.asarray(arr, dtype=np.int64), hist_max)
        counts += np.bincount(a, minlength=hist_max + 1)[:hist_max + 1]
    return counts


# Exact integer bins where the mass is (1..8), then octaves out to the vocabulary,
# so the recovered long tail stays visible instead of piling into one bar.
DEFAULT_SIZE_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 33, 65, 129, 257, 513, 1025,
                      2049, 4097, 8193, 16385, 32769, 65537, 131073, 262145]


def size_distribution(df, edges=None):
    """Heavy-tail-aware nucleus-size histogram. Returns ``(labels, counts)`` where bin
    ``i`` covers sizes ``[edges[i], edges[i+1])``. The default ``DEFAULT_SIZE_EDGES``
    keeps single-token resolution for sizes 1..8 (the singleton bar plus the small
    branches that carry the mass) and then doubles — ``9-16``, ``17-32`` … up to the
    vocabulary — so flat-distribution nuclei in the hundreds-to-tens-of-thousands show
    up as their own bars rather than a misleading spike at the cap.

    Plot ``counts`` against ``labels`` as *categorical* (equal-width) bars; an octave
    axis drawn on a linear scale would squash the head."""
    import numpy as np
    edges = np.asarray(DEFAULT_SIZE_EDGES if edges is None else edges)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for arr in df["nuc_sizes"]:
        a = np.asarray(arr, dtype=np.int64)
        counts += np.histogram(a, bins=edges)[0]
    labels = [str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}"
              for lo, hi in zip(edges[:-1], edges[1:])]
    return labels, counts


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


def size_by_position(df, max_pos: int | None = None, hist_max: int = SIZE_HIST_CAP):
    """Per-position nucleus profile across rollouts. Returns ``(mean_size,
    singleton_frac, count)`` numpy arrays indexed by token position (0 = first
    generated token). ``count[i]`` is how many rollouts reached position ``i``, so
    the deep tail is averaged over fewer, longer rollouts. Sizes are folded into
    ``[1, hist_max]`` first, so a single flat/overflowed position can't dominate a
    position's mean (a robustness choice for the positional profile)."""
    import numpy as np
    if max_pos is None:
        max_pos = max((len(a) for a in df["nuc_sizes"]), default=0)
    sum_size = np.zeros(max_pos, dtype=np.int64)
    sum_singleton = np.zeros(max_pos, dtype=np.int64)
    count = np.zeros(max_pos, dtype=np.int64)
    for arr in df["nuc_sizes"]:
        a = _fold_sizes(np.asarray(arr, dtype=np.int64), hist_max)
        n = min(len(a), max_pos)
        sum_size[:n] += a[:n]
        sum_singleton[:n] += (a[:n] == 1)
        count[:n] += 1
    denom = np.maximum(count, 1)
    return sum_size / denom, sum_singleton / denom, count


def _band_stats(counts, n_top1, sum_size_true):
    """Headline numbers for one size histogram + its top-1 count. ``sum_size_true`` is
    the slice's unfolded size sum, so ``mean_size`` reflects true nucleus sizes."""
    n_tokens = int(counts.sum())
    if not n_tokens:
        return None
    return {
        "n_tokens": n_tokens,
        "singleton_count": int(counts[1]),
        "singleton_frac": float(counts[1] / n_tokens),
        "mean_size": float(sum_size_true / n_tokens),
        "chosen_is_top1_frac": float(n_top1 / n_tokens),
    }


def summarize_nuclei(df, *, hist_max: int = SIZE_HIST_CAP, band_map: dict | None = None) -> dict:
    """Aggregate size statistics over a shard-row DataFrame.

    Reproduces the dict ``build_token_nuclei`` used to compute in its loop, but
    post-hoc from stored shards so it can run on any (e.g. subsampled) slice.

    The linear ``size_histogram`` folds sizes ``>= hist_max`` into the top bin (a
    DISPLAY bound), but ``mean_size`` is computed from the true, unfolded sizes so the
    heavy tail isn't silently clipped. ``median_size``/``p90_size`` are read off the
    folded histogram and so saturate at ``hist_max`` on very flat distributions — use
    ``size_distribution`` for the full tail.

    ``band_map`` maps ``unique_id -> band`` (e.g. from
    ``analysis.difficulty.attach_bands``/``_band_map``); when given, a per-band
    breakdown is added under ``by_band``.
    """
    import numpy as np

    ramp = np.arange(hist_max + 1)
    total = np.zeros(hist_max + 1, dtype=np.int64)
    first = np.zeros(hist_max + 1, dtype=np.int64)
    corr = np.zeros(hist_max + 1, dtype=np.int64)
    incorr = np.zeros(hist_max + 1, dtype=np.int64)
    n_top1 = n_top1_corr = n_top1_incorr = 0
    n_rollouts = 0
    sum_true = sum_true_corr = sum_true_incorr = 0   # unfolded size sums -> true means

    bands = sorted(set(band_map.values())) if band_map else []
    band_counts = {b: np.zeros(hist_max + 1, dtype=np.int64) for b in bands}
    band_top1 = {b: 0 for b in bands}
    band_sum_true = {b: 0 for b in bands}

    for row in df.itertuples(index=False):
        a = np.asarray(row.nuc_sizes, dtype=np.int64)
        if a.size == 0:
            continue
        row_true = int(np.maximum(a, 1).sum())     # true (unfolded) size sum for this row
        a = _fold_sizes(a, hist_max)
        bc = np.bincount(a, minlength=hist_max + 1)[:hist_max + 1]
        t1 = int(np.asarray(row.chosen_is_top1, dtype=bool).sum())
        total += bc
        first[a[0]] += 1
        n_top1 += t1
        n_rollouts += 1
        sum_true += row_true
        if row.answer_matches:
            corr += bc
            n_top1_corr += t1
            sum_true_corr += row_true
        else:
            incorr += bc
            n_top1_incorr += t1
            sum_true_incorr += row_true
        if band_map is not None:
            b = band_map.get(row.unique_id, "Unknown")
            if b not in band_counts:
                band_counts[b] = np.zeros(hist_max + 1, dtype=np.int64)
                band_top1[b] = 0
                band_sum_true[b] = 0
            band_counts[b] += bc
            band_top1[b] += t1
            band_sum_true[b] += row_true

    n_tokens = int(total.sum())
    frac1 = lambda c: float(c[1] / c.sum()) if c.sum() else float("nan")
    mean = lambda s: float(s / n_tokens) if n_tokens else float("nan")
    stats = {
        "n_rollouts": n_rollouts,
        "n_tokens": n_tokens,
        "singleton_count": int(total[1]),
        "singleton_frac": float(total[1] / n_tokens) if n_tokens else float("nan"),
        "mean_size": mean(sum_true),
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
            if (s := _band_stats(band_counts[b], band_top1[b], band_sum_true[b])) is not None
        }
    return stats
