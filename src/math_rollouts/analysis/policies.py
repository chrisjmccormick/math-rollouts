"""Opener-policy accuracy tables (generalizes the legacy ``openings_k16`` phase 3).

Given a nuclei table (one row per opener, with ``nuc_prob``) and a scored-rollouts
table (joined raw rollouts + scores), compute per-opener accuracy and summarize the
base model's accuracy under four first-fork opening policies:

  probability   q proportional to the model's opener prob (the natural policy)
  uniform       q uniform over openers
  acc_weighted  q proportional to observed opener accuracy
  oracle        always pick the best opener

Openers are identified by ``branch_path`` (the canonical, depth-safe key), so this
works unchanged for depth-N trees. Accuracy per opener = mean ``is_correct`` over
its rollout group; the denominator is the group's row count.
"""
from __future__ import annotations

POLICIES = ["probability", "uniform", "acc_weighted", "oracle"]


def _bp_key(x):
    return tuple(x) if x is not None else ()


def per_opener_accuracy(scored_rollouts):
    """Series indexed by (unique_id, branch_path-tuple) -> mean is_correct."""
    df = scored_rollouts.copy()
    df["_bp"] = df.branch_path.map(_bp_key)
    return df.groupby(["unique_id", "_bp"]).is_correct.mean()


def missing_openers(nuclei, acc):
    """Openers in ``nuclei`` with NO scored rollouts (their accuracy would default
    to 0). Returns ``[(unique_id, branch_path-tuple), ...]``.

    Not an error: partial opener coverage is sometimes deliberate (e.g. capping
    forced rollouts to reign in compute). ``policy_table`` calls this to WARN rather
    than silently fold absent openers in as zeros."""
    have = set(acc.index)
    return [(uid, _bp_key(bp)) for uid, g in nuclei.groupby("unique_id")
            for bp in g.branch_path if (uid, _bp_key(bp)) not in have]


def warn_missing_openers(nuclei, acc) -> list:
    """Print a one-line coverage warning if any opener lacks rollouts; return the
    missing list (empty when coverage is complete)."""
    missing = missing_openers(nuclei, acc)
    if missing:
        n_problems = len({uid for uid, _ in missing})
        examples = ", ".join(f"{uid}{list(bp)}" for uid, bp in missing[:3])
        print(f"[policies] WARNING: {len(missing)} opener(s) across {n_problems} "
              f"problem(s) have NO scored rollouts and count as accuracy 0 "
              f"(e.g. {examples}). This is expected only if opener coverage was "
              f"deliberately capped.", flush=True)
    return missing


def policy_table(nuclei, scored_rollouts):
    """One row per problem with accuracy under each policy. ``nuclei`` needs
    columns unique_id, subject, branch_path, nuc_prob.

    Runs an opener-coverage pre-pass first: openers with no scored rollouts default
    to accuracy 0, so they are surfaced via a warning rather than folded in
    silently (see ``warn_missing_openers``)."""
    import numpy as np
    import pandas as pd

    acc = per_opener_accuracy(scored_rollouts)
    warn_missing_openers(nuclei, acc)
    rows = []
    for uid, g in nuclei.groupby("unique_id"):
        a = np.array([acc.get((uid, _bp_key(bp)), 0.0) for bp in g.branch_path])
        p = np.asarray(g.nuc_prob, dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.full(len(a), 1.0 / max(len(a), 1))
        rows.append(dict(
            unique_id=uid, subject=g.subject.iloc[0], n_openers=len(a),
            probability=float((p * a).sum()),
            uniform=float(a.mean()) if len(a) else 0.0,
            acc_weighted=float((a * a).sum() / a.sum()) if a.sum() > 0 else float(a.mean()),
            oracle=float(a.max()) if len(a) else 0.0,
        ))
    return pd.DataFrame(rows)


def summarize(policy_df) -> str:
    """Subject x policy accuracy table as a printable string."""
    lines = [f"{'subject':<26}{'n':>4}  " + "".join(f"{c:>14}" for c in POLICIES)]
    for subj in sorted(policy_df.subject.unique()) + ["ALL"]:
        sub = policy_df if subj == "ALL" else policy_df[policy_df.subject == subj]
        lines.append(f"{subj:<26}{len(sub):>4}  "
                     + "".join(f"{sub[c].mean()*100:12.1f}% " for c in POLICIES))
    return "\n".join(lines)


def join_scores(rollouts, scores):
    """Left-join RAW rollouts with DERIVED scores on the rollout key.

    ``branch_path`` is compared by value (as a tuple), since list columns are not
    directly joinable. Returns the rollouts frame with the score columns
    (``is_correct`` etc.) attached — the input that ``policy_table`` expects."""
    from ..schema import ROLLOUT_KEY
    r, s = rollouts.copy(), scores.copy()
    for df in (r, s):
        df["_bp"] = df.branch_path.map(_bp_key)
    keys = [k for k in ROLLOUT_KEY if k != "branch_path"] + ["_bp"]
    merged = r.merge(s.drop(columns=["branch_path"]), on=keys, how="left",
                     suffixes=("", "_score"))
    return merged.drop(columns=["_bp"])


def main() -> None:
    """CLI (installed as ``math-rollouts-policies``): compute ``policies.csv`` for an
    opener experiment dir holding nuclei + raw rollouts + derived scores.

    Keeps the opener-policy summary a DERIVED analysis artifact (joins rollouts x
    scores on the rollout key, groups per opener by ``branch_path``), recomputable
    under any ``scorer_id`` without touching generation."""
    import argparse
    from pathlib import Path

    import pandas as pd

    ap = argparse.ArgumentParser(description="opener-policy accuracy table (DERIVED)")
    ap.add_argument("--exp-dir", required=True,
                    help="experiment dir with nuclei/rollouts/scores parquet")
    ap.add_argument("--scorer", default=None, help="filter scores to this scorer_id")
    ap.add_argument("--out", default=None, help="output csv (default: <exp-dir>/policies.csv)")
    a = ap.parse_args()

    d = Path(a.exp_dir)
    nuclei = pd.read_parquet(d / "nuclei.parquet")
    rollouts = pd.read_parquet(d / "rollouts.parquet")
    scores = pd.read_parquet(d / "scores.parquet")
    if a.scorer:
        scores = scores[scores.scorer_id == a.scorer]

    table = policy_table(nuclei, join_scores(rollouts, scores))
    out = Path(a.out) if a.out else d / "policies.csv"
    table.to_csv(out, index=False)
    print(summarize(table))
    print(f"\nwrote {out}  ({len(table)} problems)", flush=True)


if __name__ == "__main__":
    main()
