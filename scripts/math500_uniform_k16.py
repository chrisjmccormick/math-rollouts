#!/usr/bin/env python3
"""MATH-500 uniform-opener generation job (depth-1 first-token nucleus).

Depth-1 nucleus tree (= classic first-token nucleus), K=16 forced uniform rollouts
through every opener, all 500 problems. Writes RAW nuclei + rollouts + manifest into
the local dataset layout; scoring is a SEPARATE pass.

Defaults to base Qwen2.5-Math-1.5B (the regenerated job that recovers the lost
openers); pass ``--model`` for any registered non-thinking/thinking checkpoint, e.g.
``sail/Qwen2.5-Math-1.5B-Oat-Zero``. The output dir slug follows the model id, so
runs land side by side (``generations/<slug>/math500_uniform_k16_d1/``).

NOTE on Oat-Zero: its first-token distribution is sharply peaked (opener "To"
typically carries > top_p), so most problems yield a SINGLETON nucleus — one opener
per problem, branch_size 1, all four opener policies collapse to the same number.
That collapse is itself the thing nucleus-viz renders; the data captures it via
``branch_size`` / ``nuc_prob``.

Run on the A100 in the project env::

    source ~/env.sh && python scripts/math500_uniform_k16.py --out-root /path/to/data
    source ~/env.sh && python scripts/math500_uniform_k16.py --out-root /path/to/data \\
        --model sail/Qwen2.5-Math-1.5B-Oat-Zero

Then score (CPU)::

    math-rollouts-score --rollouts <out-root>/generations/<slug>/math500_uniform_k16_d1/rollouts.parquet
"""
from __future__ import annotations

import argparse

from math_rollouts.config import GenConfig
from math_rollouts.generate.run import generate

MODEL = "Qwen/Qwen2.5-Math-1.5B"
EXPERIMENT = "math500_uniform_k16_d1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True,
                    help="local dataset root (output -> generations/<model-slug>/...)")
    ap.add_argument("--model", default=MODEL, dest="model_id",
                    help=f"registered checkpoint (default: {MODEL})")
    ap.add_argument("--experiment", default=EXPERIMENT,
                    help=f"experiment dir name (default: {EXPERIMENT})")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = generate(
        model_id=a.model_id, experiment=a.experiment, out_root=a.out_root,
        k=a.k, max_depth=1, max_branch=None, run_id=a.run_id, seed=a.seed,
        coverage="math500", ids=None, cfg=GenConfig(), device=a.device,
    )
    print(f"[math500_uniform_k16] {a.model_id} done -> {out}", flush=True)


if __name__ == "__main__":
    main()
