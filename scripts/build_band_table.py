#!/usr/bin/env python3
"""Thin wrapper for the head-to-head banded comparison table.

Implementation lives in the package (``math_rollouts.analysis.bandtable:main``),
also installed as the console script ``math-rollouts-bandtable``.

    python scripts/build_band_table.py --band-model Qwen/Qwen2.5-Math-1.5B \\
        --cohort Base=<...>/math500_passK.parquet \\
        --cohort Oat-Zero=<...>/oat_math500_passK.parquet --out table.md
"""
from math_rollouts.analysis.bandtable import main

if __name__ == "__main__":
    main()
