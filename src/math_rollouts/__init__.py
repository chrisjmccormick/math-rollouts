"""math_rollouts — naturally-sampled math rollouts + uniform-opener/branch nuclei.

Guidance-free, base/naturally-sampled artifacts only (no teacher-guided
intersection sampling, no SFT-model rollouts). The generation code is unified on
the anytree nucleus-tree methodology (depth-1 first-token nucleus is a special
case of a depth-N branch tree) and supports both thinking and non-thinking models
through a single ModelAdapter abstraction.

Generation and scoring are deliberately separated: generation (GPU) writes raw
rollouts; a re-runnable scoring pass (CPU) writes derived scores. Raw rollouts are
the durable source of truth so accuracy can be recomputed under a different scorer.
"""

__version__ = "0.1.0"
