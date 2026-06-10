#!/usr/bin/env python3
"""One-off: re-key ``unique_id`` to embed the split, and drop the second id.

Before: ``unique_id`` lied — every math12k row was ``train/<subj>/<n>`` regardless
of split, and math500 rows carried a second id (``math500_native_id``, the HF
MATH-500 ``test/<subj>/<n>.json``). Worse, the per-problem experiment files +
``policies.csv`` keyed on the HF id while the pools keyed on the math12k id, so the
SAME problem had two ids across files.

After: a single ``unique_id`` per problem, ``<split>/<subj>/<n>`` with split in
{train, test, math500} (authoritative ``split`` column from math_problems) and the
stable math12k number. ``math500_native_id`` is dropped from the bulk rows; the
HF MATH-500 cross-ref lives once in ``mappings/math500_to_hf.csv`` (and stays on
``problems/math500.parquet``).

The remap is total over both source formats::

    train/precalculus/12414      -> math500/precalculus/12414   (split swap)
    test/precalculus/807.json    -> math500/precalculus/12414   (HF id -> math12k twin)

so the pool and the experiment versions of a math500 problem collapse to one id.
Streams parquets batch-wise (the big pools have huge list columns), copies
manifests untouched, replaces the old ``mappings/math500_to_math12k.*``.

Usage::

    python scripts/migrate_unique_id_splits.py --out-root /path/to/migrated [--in-root SNAP]

If ``--in-root`` is omitted the dataset is snapshot-downloaded first.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "ChrisMcCormick/math-rollouts"
OLD_MAPPING_PREFIX = "mappings/math500_to_math12k"
NEW_MAPPING = "mappings/math500_to_hf.csv"


def _new_from_math12k(old: str, split: str) -> str:
    """``train/geometry/9467`` + split ``math500`` -> ``math500/geometry/9467``."""
    return "/".join([split, *old.split("/")[1:]])


def build_maps(mp: pd.DataFrame, m500: pd.DataFrame):
    """Combined map old_id -> new_id (both train/... and test/...json sources), plus
    the new HF cross-ref rows (new_id, hf_math500_id). ``mp``/``m500`` are the
    ``problems/math_problems.parquet`` / ``problems/math500.parquet`` tables.

    Works against the tables before OR after their own re-key: every problem is also
    registered under its legacy math12k alias ``train/<subj>/<n>`` (every math12k row
    was ``train/...`` regardless of split), and an already-new id maps to itself —
    so the map stays total over all three source formats."""
    split_map = {}
    for u, s in zip(mp.unique_id, mp.split):
        new = _new_from_math12k(u, s)
        split_map[u] = new
        split_map["/".join(["train", *u.split("/")[1:]])] = new

    hf_to_new, cross = {}, []
    for train_id, hf_id in zip(m500.unique_id, m500.math500_native_id):
        new_id = split_map[train_id]
        hf_to_new[hf_id] = new_id
        cross.append((new_id, hf_id))
    combined = {**split_map, **hf_to_new}
    return combined, pd.DataFrame(cross, columns=["unique_id", "hf_math500_id"])


def remap_parquet(in_path: Path, out_path: Path, cmap: dict, *, drop_native: bool,
                  batch_size: int = 20000) -> tuple[int, int]:
    """Stream-rewrite a parquet: remap ``unique_id``, optionally drop
    ``math500_native_id``. Returns (rows_in, distinct unmapped)."""
    pf = pq.ParquetFile(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    unmapped: set[str] = set()
    for batch in pf.iter_batches(batch_size=batch_size):
        t = pa.Table.from_batches([batch])
        uids = t.column("unique_id").to_pylist()
        new = []
        for u in uids:
            n = cmap.get(u)
            if n is None:
                unmapped.add(u)
                n = u
            new.append(n)
        t = t.set_column(t.schema.get_field_index("unique_id"), "unique_id",
                         pa.array(new, type=pa.string()))
        if drop_native and "math500_native_id" in t.schema.names:
            t = t.drop(["math500_native_id"])
        if writer is None:
            writer = pq.ParquetWriter(out_path, t.schema, compression="zstd")
        writer.write_table(t)
        rows += t.num_rows
    if writer is not None:
        writer.close()
    return rows, len(unmapped)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--in-root", type=Path, default=None,
                    help="local snapshot of the dataset; if omitted, download it")
    a = ap.parse_args()

    in_root = a.in_root
    if in_root is None:
        from huggingface_hub import snapshot_download
        in_root = Path(snapshot_download(REPO, repo_type="dataset"))
        print(f"snapshot at {in_root}")
    out_root = a.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    cmap, cross = build_maps(pd.read_parquet(in_root / "problems" / "math_problems.parquet"),
                             pd.read_parquet(in_root / "problems" / "math500.parquet"))
    print(f"map covers {len(cmap)} source ids "
          f"({sum(v.startswith('math500/') for v in set(cmap.values()))} math500)")

    total_unmapped = 0
    for src in sorted(in_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(in_root).as_posix()
        if rel.startswith(".") or "/.cache" in rel or rel.endswith(".lock"):
            continue
        if rel.startswith(OLD_MAPPING_PREFIX):
            continue                                   # replaced by NEW_MAPPING
        dst = out_root / rel
        if rel.endswith(".parquet"):
            drop = rel != "problems/math500.parquet"   # keep the cross-ref on math500
            rin, un = remap_parquet(src, dst, cmap, drop_native=drop)
            rout = pq.ParquetFile(dst).metadata.num_rows
            flag = "  !! UNMAPPED" if un else ""
            assert rin == rout, f"row mismatch {rel}: {rin} != {rout}"
            total_unmapped += un
            print(f"  {rel}: {rin} rows{flag}")
        elif rel == "generations/qwen2.5-math-1.5b/math500_uniform_k16_d1/policies.csv" \
                or rel.endswith("policies.csv"):
            df = pd.read_csv(src)
            un = sorted(set(df.unique_id) - set(cmap))
            df["unique_id"] = df.unique_id.map(lambda u: cmap.get(u, u))
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst, index=False)
            total_unmapped += len(un)
            print(f"  {rel}: {len(df)} rows{'  !! UNMAPPED' if un else ''}")
        else:                                          # manifests etc. — copy
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    (out_root / "mappings").mkdir(parents=True, exist_ok=True)
    cross.sort_values("unique_id").to_csv(out_root / NEW_MAPPING, index=False)
    print(f"wrote {NEW_MAPPING} ({len(cross)} rows)")
    print(f"\nDONE -> {out_root}   (total unmapped ids: {total_unmapped})")
    if total_unmapped:
        raise SystemExit("ABORT: some ids were not in the map (left unchanged)")


if __name__ == "__main__":
    main()
