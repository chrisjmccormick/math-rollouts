#!/usr/bin/env python3
"""Thin wrapper for the opener-policy table.

The implementation lives in the package (``math_rollouts.analysis.policies:main``),
also installed as the console script ``math-rollouts-policies``. This shim is kept
so the repo's ``scripts/`` entry points stay runnable without an install.

    python scripts/build_policies.py --exp-dir <root>/generations/<slug>/<exp>
"""
from math_rollouts.analysis.policies import main

if __name__ == "__main__":
    main()
