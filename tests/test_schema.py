"""Schema dtype + row-builder + key-constant validation."""
from __future__ import annotations

import pyarrow as pa

from math_rollouts.schema import (
    GROUP_KEY,
    NUCLEI_SCHEMA,
    ROLLOUT_KEY,
    ROLLOUTS_SCHEMA,
    SCORES_SCHEMA,
    table_from_rows,
)


def _dtype(schema, name):
    return schema.field(name).type


def test_nuclei_dtypes():
    assert _dtype(NUCLEI_SCHEMA, "depth") == pa.int8()
    assert _dtype(NUCLEI_SCHEMA, "branch_path") == pa.list_(pa.int16())
    assert _dtype(NUCLEI_SCHEMA, "opener_token_ids") == pa.list_(pa.int32())
    assert _dtype(NUCLEI_SCHEMA, "opener_token_strs") == pa.list_(pa.string())
    assert _dtype(NUCLEI_SCHEMA, "is_thinking") == pa.bool_()


def test_rollouts_has_no_correctness():
    # RAW table must NOT carry derived correctness — that lives in scores.parquet.
    names = set(ROLLOUTS_SCHEMA.names)
    assert "is_correct" not in names
    assert {"run_id", "gen_config_id", "seed", "sample_idx"} <= names
    assert _dtype(ROLLOUTS_SCHEMA, "completion_token_ids") == pa.list_(pa.int32())


def test_single_split_aware_id():
    # The second id is gone; unique_id (split-aware) is the only problem identity.
    assert "math500_native_id" not in NUCLEI_SCHEMA.names
    assert "math500_native_id" not in ROLLOUTS_SCHEMA.names
    assert "unique_id" in NUCLEI_SCHEMA.names and "unique_id" in ROLLOUTS_SCHEMA.names


def test_scores_dtypes_and_join_keys():
    assert _dtype(SCORES_SCHEMA, "is_correct") == pa.bool_()
    assert ROLLOUT_KEY == ["model_id", "unique_id", "run_id", "branch_path", "sample_idx"]
    assert GROUP_KEY == ["model_id", "unique_id", "branch_path", "run_id"]
    # every join/group key must exist in the tables that need it.
    for k in ROLLOUT_KEY:
        assert k in SCORES_SCHEMA.names and k in ROLLOUTS_SCHEMA.names
    for k in GROUP_KEY:
        assert k in ROLLOUTS_SCHEMA.names


def test_table_from_rows_coerces_and_fills_missing():
    rows = [{
        "model_id": "M", "unique_id": "math500/geometry/9467", "subject": "Geometry",
        "answer": "42", "depth": 1, "branch_path": [3],
        "opener_token_ids": [123], "opener_token_strs": ["x"],
        "fork_token_id": 123, "nuc_prob": 0.5, "path_prob": 0.5,
        "branch_size": 7, "terminal": None, "is_thinking": False,
    }]
    tbl = table_from_rows(rows, NUCLEI_SCHEMA)
    assert tbl.schema.equals(NUCLEI_SCHEMA)
    assert tbl.num_rows == 1
    # a missing column is filled with null rather than raising.
    tbl2 = table_from_rows([{"model_id": "M"}], NUCLEI_SCHEMA)
    assert tbl2.column("unique_id")[0].as_py() is None
