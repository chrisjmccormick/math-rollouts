"""Head-to-head banded comparison table (generalizes the legacy ``math500_band_table.py``).

Compares N model cohorts on a shared problem set: rows grouped by difficulty band
(from :mod:`math_rollouts.analysis.difficulty`) and broken out by subject. Each cell
is the POOLED tally over that band x subject cell — ``Σ n_correct / Σ n`` across
``problems x K`` rollouts, NOT a per-problem average — with the best value in the row
bolded; when exactly two cohorts are compared a ``Δ`` (first − second) column is added.

De-hardcoded and model-agnostic. The original script pinned an in-house SFT cohort and
an Oat-Zero cohort with absolute paths; nothing here is tied to a particular model —
pass any labeled cohorts and any ``band_model`` whose solve-rate defines the bands.
The banding itself is reused from ``difficulty.band_table`` (no duplicated thresholds).

    from math_rollouts.analysis.bandtable import per_problem, band_compare, render_markdown
    from math_rollouts.analysis.difficulty import band_table
    bands = dict(zip(bt := band_table("Qwen/Qwen2.5-Math-1.5B"), ...))  # uid -> band
    tidy = band_compare([("Base", per_problem(base_df)), ("Oat-Zero", per_problem(oat_df))], bands)
    print(render_markdown(tidy, ["Base", "Oat-Zero"]))
"""
from __future__ import annotations

from .difficulty import BAND_ORDER

UNKNOWN_BAND = "Unknown"


def per_problem(df, *, id_col: str = "unique_id", correct_col: str = "is_correct",
                run_ids=None):
    """Per-problem integer tallies for one cohort: ``id_col, subject, n_correct, n``.

    ``n`` is the rollout count (the group denominator) and ``n_correct`` the number
    of correct rollouts — both summed, never averaged, so cells pool cleanly.

    Batch hygiene: cells are pooled over ``problems x K``, so pooling rollouts from
    runs with *different* sample budgets (e.g. a uniform baseline batch + a pass@k
    expansion that oversamples hard problems) silently K-weights the result. This
    function therefore refuses to guess: if ``df`` spans more than one ``run_id`` you
    must name the batch(es) via ``run_ids`` (a single id, or a list to pool several
    equivalent runs deliberately). Selecting is the caller's responsibility — nothing
    here designates a "canonical" run."""
    if "run_id" in df.columns:
        present = sorted(df["run_id"].dropna().unique().tolist())
        if run_ids is not None:
            want = {run_ids} if not isinstance(run_ids, (list, tuple, set)) else set(run_ids)
            df = df[df["run_id"].isin(want)]
        elif len(present) > 1:
            raise ValueError(
                f"cohort spans run_ids {present} which may carry different sample "
                f"budgets per problem; pooling them would weight the table by rollout "
                f"count. Pass run_ids=<id> (or a list to pool runs deliberately) — the "
                f"caller owns batch selection."
            )
    g = df.groupby([id_col, "subject"])[correct_col]
    out = g.agg(n_correct="sum", n="count").reset_index()
    out["n_correct"] = out["n_correct"].astype(int)
    out["n"] = out["n"].astype(int)
    return out


def band_compare(cohorts, band_lookup, *, id_col: str = "unique_id"):
    """Stack per-cohort per-problem tallies into one tidy long frame.

    ``cohorts``: list of ``(label, per_problem_df)``. ``band_lookup``: mapping
    ``unique_id -> band`` (e.g. ``dict(zip(bt.unique_id, bt.band))`` from
    ``difficulty.band_table``). Problems with no band map to ``"Unknown"``.

    Returns columns: ``band, subject, label, unique_id, n_correct, n``."""
    import pandas as pd

    frames = []
    for label, pp in cohorts:
        t = pp.copy()
        t["label"] = label
        t["band"] = t[id_col].map(band_lookup).fillna(UNKNOWN_BAND)
        frames.append(t.rename(columns={id_col: "unique_id"})[
            ["band", "subject", "label", "unique_id", "n_correct", "n"]])
    return pd.concat(frames, ignore_index=True)


def _pool(sub, label):
    g = sub[sub.label == label]
    nc, n = int(g.n_correct.sum()), int(g.n.sum())
    return nc, n, (100.0 * nc / n if n else 0.0)


def _fmt_cell(nc: int, n: int, bold: bool) -> str:
    pct = 100.0 * nc / n if n else 0.0
    s = f"({nc} / {n}) {pct:.1f}%"
    return f"**{s}**" if bold else s


def _render_row(row_label, sub, labels, delta, emph):
    pools = {lab: _pool(sub, lab) for lab in labels}
    best = max((p[2] for p in pools.values() if p[1] > 0), default=None)
    cells = [emph(_fmt_cell(nc, n, best is not None and pct == best and n > 0))
             for nc, n, pct in (pools[lab] for lab in labels)]
    if delta:
        a, b = delta
        d = pools[a][2] - pools[b][2]
        cells.append(emph(f"{'+' if d >= 0 else ''}{d:.1f}pp"))
    return f"| {row_label} | " + " | ".join(cells) + " |"


def render_markdown(tidy, labels, *, band_order=BAND_ORDER, title: str | None = None,
                    delta="auto") -> str:
    """Render the tidy frame from :func:`band_compare` as a markdown table.

    ``labels`` fixes the column order. ``delta``: ``(a, b)`` adds a ``Δ (a − b)``
    column; ``"auto"`` (default) adds it iff exactly two cohorts are compared; falsy
    disables it. Bands are emitted in ``band_order`` (then any extras, e.g.
    ``"Unknown"``, alphabetically). Each band shows per-subject rows, a band subtotal
    (italic) and an overall row (bold)."""
    if delta == "auto":
        delta = (labels[0], labels[1]) if len(labels) == 2 else None

    n_cols = len(labels) + (1 if delta else 0)
    lines = []
    if title:
        lines += [title, ""]
    header = "| Subject | " + " | ".join(labels)
    if delta:
        header += f" | Δ ({delta[0]} − {delta[1]})"
    lines.append(header + " |")
    lines.append("|" + "---|" * (n_cols + 1))

    present = [b for b in band_order if (tidy.band == b).any()]
    extra = sorted(b for b in tidy.band.unique() if b not in band_order)
    for band in present + extra:
        bsub = tidy[tidy.band == band]
        n_problems = bsub.unique_id.nunique()
        lines.append(f"| **{band}** ({n_problems} problems) |" + " |" * n_cols)
        for subject, grp in sorted(bsub.groupby("subject"), key=lambda x: x[0]):
            lines.append(_render_row(subject, grp, labels, delta, emph=lambda s: s))
        lines.append(_render_row(f"*{band} total*", bsub, labels, delta,
                                 emph=lambda s: f"*{s}*"))
    lines.append(_render_row("**Overall**", tidy, labels, delta,
                             emph=lambda s: f"**{s}**"))
    return "\n".join(lines)


def main() -> None:
    """CLI (installed as ``math-rollouts-bandtable``): markdown head-to-head table.

    Bands by ``--band-model`` (resolved through ``difficulty.band_table`` against the
    dataset's natural-gen pools) and compares one or more ``--cohort LABEL=PATH``
    parquets (each carrying ``is_correct``, ``unique_id``, ``subject``)."""
    import argparse
    from pathlib import Path

    import pandas as pd

    from .difficulty import band_table

    ap = argparse.ArgumentParser(description="head-to-head banded comparison (markdown)")
    ap.add_argument("--band-model", required=True,
                    help="model_id whose solve-rate defines difficulty bands")
    ap.add_argument("--cohort", action="append", required=True,
                    metavar="LABEL=PATH[#RUN_ID[,RUN_ID...]]",
                    help="labeled parquet (is_correct+unique_id+subject); repeatable. "
                         "Append #RUN_ID to select a batch when the file bundles several "
                         "(e.g. a baseline run + a pass@k expansion).")
    ap.add_argument("--data-root", default=None, help="local dataset root for banding")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="write markdown here (also printed)")
    a = ap.parse_args()

    bt = band_table(a.band_model, a.data_root)
    band_lookup = dict(zip(bt.unique_id, bt.band))

    def _coerce(r):
        r = r.strip()
        return int(r) if r.lstrip("-").isdigit() else r

    cohorts, labels = [], []
    for spec in a.cohort:
        label, _, rhs = spec.partition("=")
        if not rhs:
            ap.error(f"--cohort expects LABEL=PATH[#RUN_ID], got {spec!r}")
        path, sep, ridstr = rhs.partition("#")
        run_ids = [_coerce(r) for r in ridstr.split(",")] if sep else None
        try:
            cohorts.append((label, per_problem(pd.read_parquet(path), run_ids=run_ids)))
        except ValueError as e:
            ap.error(f"cohort {label!r}: {e}  "
                     f"e.g. --cohort '{label}={path}#<run_id>'")
        labels.append(label)

    tidy = band_compare(cohorts, band_lookup)
    md = render_markdown(tidy, labels, title=a.title)
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n")
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
