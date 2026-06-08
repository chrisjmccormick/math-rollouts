"""Per-problem difficulty banding by base-model solve rate.

A problem's difficulty is the base model's empirical solve rate (mean ``is_correct``
over its naturally-sampled rollouts), bucketed into bands. Ported from the source
``difficulty.py`` but DE-HARDCODED: instead of literal absolute paths, the
generation parquets are resolved through the dataset's ``generations/<slug>/...``
convention (local snapshot via ``$MATH_ROLLOUTS_DATA`` or the HF hub).

Banding reads the NATURALLY-SAMPLED pools (which carry ``is_correct`` + ``unique_id``)
— e.g. ``math12k_L4_5_K64`` and ``math500_passK`` — not the opener experiments. A
problem present in more than one file is averaged.

    from math_rollouts.analysis.difficulty import band_for, band_table, BAND_ORDER
    band_for("Qwen/Qwen2.5-Math-1.5B", "math500/geometry/9467")   # -> "Medium"
"""
from __future__ import annotations

from functools import lru_cache

from ..data.hf import _resolve, model_slug

BAND_ORDER = ["Easy", "Medium", "Hard", "Very Hard", "Impossible"]

# model_id -> experiment subdirs whose rollouts define difficulty. Each resolves to
# generations/<slug>/<exp>/rollouts-or-legacy.parquet. The migrated natural-gen
# cohorts keep the legacy schema (is_correct present), so a plain parquet name is
# used here. Add models by adding a line — no other code changes.
MODEL_DATA = {
    "Qwen/Qwen2.5-Math-1.5B": ["math12k_L4_5_K64", "math500_passK"],
    "sail/Qwen2.5-Math-1.5B-Oat-Zero": ["math12k_K64", "math500_passK"],
}


def assign_band(acc: float) -> str:
    if acc == 0.0:
        return "Impossible"
    elif acc < 0.25:
        return "Very Hard"
    elif acc < 0.50:
        return "Hard"
    elif acc < 0.75:
        return "Medium"
    return "Easy"


def _gen_paths(model_id: str, data_root=None):
    slug = model_slug(model_id)
    for exp in MODEL_DATA[model_id]:
        yield _resolve(f"generations/{slug}/{exp}.parquet", local_root=data_root)


@lru_cache(maxsize=8)
def band_table(model: str, data_root: str | None = None):
    """Per-problem difficulty: columns unique_id, n, acc, band.

    Pools rollouts across the model's data files and groups by the split-aware
    ``unique_id`` (e.g. ``math500/geometry/9467``)."""
    import pandas as pd

    if model not in MODEL_DATA:
        raise KeyError(f"no difficulty data registered for {model!r}; known: {sorted(MODEL_DATA)}")
    frames = [pd.read_parquet(path, columns=["unique_id", "is_correct"])
              for path in _gen_paths(model, data_root)]
    allrows = pd.concat(frames, ignore_index=True)

    out = allrows.groupby("unique_id").is_correct.agg(["mean", "size"]).reset_index()
    out.columns = ["unique_id", "acc", "n"]
    out["band"] = out["acc"].apply(assign_band)
    return out


@lru_cache(maxsize=8)
def _band_map(model: str, data_root: str | None = None) -> dict:
    t = band_table(model, data_root)
    return dict(zip(t.unique_id, t.band))


@lru_cache(maxsize=8)
def _acc_map(model: str, data_root: str | None = None) -> dict:
    t = band_table(model, data_root)
    return dict(zip(t.unique_id, t.acc))


def band_for(model: str, unique_id: str, default: str = "Unknown", data_root: str | None = None) -> str:
    return _band_map(model, data_root).get(unique_id, default)


def acc_for(model: str, unique_id: str, default=None, data_root: str | None = None):
    return _acc_map(model, data_root).get(unique_id, default)


def attach_bands(df, model: str, id_col: str = "unique_id", data_root: str | None = None):
    bmap, amap = _band_map(model, data_root), _acc_map(model, data_root)
    out = df.copy()
    out["acc"] = out[id_col].map(amap)
    out["band"] = out[id_col].map(bmap).fillna("Unknown")
    return out
