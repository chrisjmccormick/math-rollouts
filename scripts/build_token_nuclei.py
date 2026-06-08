#!/usr/bin/env python3
"""Per-token nucleus-size statistics over a rollout pool (e.g. math500_passK).

Teacher-forces every rollout, records the nucleus size at each generated token,
and writes ``<pool>_token_nuclei.parquet`` + ``<pool>_nuclei_stats.json`` under the
local dataset layout. The headline numbers (singleton fraction, size histogram) are
printed. Thin wrapper around ``math_rollouts.analysis.token_nuclei``.

GPU box, project env::

    python scripts/build_token_nuclei.py --out-root /path/to/data
    python scripts/build_token_nuclei.py --out-root /path/to/data \\
        --model sail/Qwen2.5-Math-1.5B-Oat-Zero --pool math500_passK
"""
from math_rollouts.analysis.token_nuclei import main

if __name__ == "__main__":
    main()
