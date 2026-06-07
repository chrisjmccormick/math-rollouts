"""Thin loaders over the ``math-rollouts`` HF dataset layout.

Layout (repo_type=dataset)::

    problems/    math_problems.parquet, math500.parquet
    mappings/    math500_to_math12k.{json,csv}
    generations/<model-slug>/<experiment>/
        nuclei.parquet, rollouts.parquet, scores.parquet, policies.csv, manifest.json

``model_slug`` = lowercase HF id minus the org (``Qwen/Qwen2.5-Math-1.5B`` ->
``qwen2.5-math-1.5b``). Consumers (e.g. the separate nucleus-viz repo) read this
layout directly; a single file is fetched via ``hf_hub_download`` and cached.
"""
from __future__ import annotations

import os
from pathlib import Path

DATASET_REPO = "ChrisMcCormick/math-rollouts"


def model_slug(model_id: str) -> str:
    """``Qwen/Qwen2.5-Math-1.5B`` -> ``qwen2.5-math-1.5b``."""
    return model_id.split("/")[-1].lower()


def _resolve(rel_path: str, *, local_root: str | Path | None = None,
             repo_id: str = DATASET_REPO, revision: str | None = None) -> str:
    """Return a local filesystem path for ``rel_path`` within the dataset, either
    from a local snapshot (``local_root`` or ``$MATH_ROLLOUTS_DATA``) or by
    downloading from the HF hub."""
    root = local_root or os.environ.get("MATH_ROLLOUTS_DATA")
    if root:
        cand = Path(root) / rel_path
        if cand.exists():
            return str(cand)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=rel_path, revision=revision,
                           repo_type="dataset")


def experiment_dir(model_id: str, experiment: str) -> str:
    return f"generations/{model_slug(model_id)}/{experiment}"


def load_nuclei(model_id: str, experiment: str, **kw):
    import pandas as pd
    return pd.read_parquet(_resolve(f"{experiment_dir(model_id, experiment)}/nuclei.parquet", **kw))


def load_rollouts(model_id: str, experiment: str, **kw):
    import pandas as pd
    return pd.read_parquet(_resolve(f"{experiment_dir(model_id, experiment)}/rollouts.parquet", **kw))


def load_scores(model_id: str, experiment: str, **kw):
    import pandas as pd
    return pd.read_parquet(_resolve(f"{experiment_dir(model_id, experiment)}/scores.parquet", **kw))


def load_scored_rollouts(model_id: str, experiment: str, scorer_id: str | None = None, **kw):
    """Raw rollouts joined to their scores on the rollout key (optionally filtered
    to one ``scorer_id``). The join consumers actually want for accuracy work."""
    from ..schema import ROLLOUT_KEY

    r = load_rollouts(model_id, experiment, **kw)
    s = load_scores(model_id, experiment, **kw)
    if scorer_id is not None:
        s = s[s.scorer_id == scorer_id]
    # branch_path is a list column; stringify for a hashable join key.
    for df in (r, s):
        df["_bp"] = df.branch_path.map(lambda x: tuple(x) if x is not None else ())
    keys = [k for k in ROLLOUT_KEY if k != "branch_path"] + ["_bp"]
    merged = r.merge(s.drop(columns=["branch_path"]), on=keys, how="left", suffixes=("", "_score"))
    return merged.drop(columns=["_bp"])
