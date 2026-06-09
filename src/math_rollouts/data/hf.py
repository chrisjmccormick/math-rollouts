"""Thin loaders over the ``math-rollouts`` HF dataset layout.

Layout (repo_type=dataset)::

    problems/    math_problems.parquet, math500.parquet
    mappings/    math500_to_hf.csv   (unique_id <-> hf MATH-500 id)
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


def load_token_nuclei(model_id: str, pool: str, unique_id: str, **kw):
    """Load one problem's per-token nucleus shard from
    ``generations/<slug>/<pool>_token_nuclei/<uid-slug>.parquet`` (per-problem
    sharding, the default). ``unique_id`` ``train/geometry/9467`` ->
    ``train_geometry_9467.parquet``. Raises if the shard isn't present (the store
    is produced by ``analysis.token_nuclei`` and may not exist yet)."""
    import pandas as pd
    slug = unique_id.replace("/", "_")
    rel = f"generations/{model_slug(model_id)}/{pool}_token_nuclei/{slug}.parquet"
    return pd.read_parquet(_resolve(rel, **kw))


def load_token_nuclei_pool(model_id: str, pool: str, *, columns: list[str] | None = None,
                           local_root: str | Path | None = None,
                           repo_id: str = DATASET_REPO, revision: str | None = None):
    """Load and concatenate ALL per-problem nucleus shards for ``pool`` into one
    DataFrame (the bulk counterpart to ``load_token_nuclei``, which reads a single
    problem). Resolves a local snapshot (``local_root`` or ``$MATH_ROLLOUTS_DATA``)
    if present, else pulls just the ``<pool>_token_nuclei/*.parquet`` shards from the
    hub via ``snapshot_download``.

    ``columns`` is forwarded to ``read_parquet`` — pass the light columns (e.g.
    ``["unique_id", "answer_matches", "nuc_sizes", "chosen_is_top1"]``) to skip the bulky
    ``kept_ids``/``kept_logits`` lists when you only need size statistics."""
    import pandas as pd

    rel_dir = f"generations/{model_slug(model_id)}/{pool}_token_nuclei"
    root = local_root or os.environ.get("MATH_ROLLOUTS_DATA")
    shard_dir = None
    if root and (Path(root) / rel_dir).is_dir():
        shard_dir = Path(root) / rel_dir
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                 allow_patterns=f"{rel_dir}/*.parquet")
        shard_dir = Path(snap) / rel_dir
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no nucleus shards under {shard_dir}")
    return pd.concat([pd.read_parquet(p, columns=columns) for p in shards],
                     ignore_index=True)


def load_problems_parquet(name: str = "math_problems", **kw):
    """Load ``problems/<name>.parquet``: ``math_problems`` (the ~12k MATH superset,
    keyed by ``train/<subj>/<n>`` ids) or ``math500`` (the 500-problem subset). Used
    to recover problem text for prompt reconstruction. Suffix optional."""
    import pandas as pd
    if not name.endswith(".parquet"):
        name += ".parquet"
    return pd.read_parquet(_resolve(f"problems/{name}", **kw))


def load_generation_parquet(model_id: str, name: str, **kw):
    """Load a standalone ``generations/<model-slug>/<name>.parquet`` (not part of an
    experiment dir), e.g. ``name="math500_passK"`` — the naturally-sampled rollout
    pools. ``name`` may include or omit the ``.parquet`` suffix."""
    import pandas as pd
    if not name.endswith(".parquet"):
        name += ".parquet"
    return pd.read_parquet(_resolve(f"generations/{model_slug(model_id)}/{name}", **kw))


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
