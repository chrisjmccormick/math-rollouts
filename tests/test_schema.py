"""Schema dtype + row-builder + key-constant validation."""
from __future__ import annotations

import pyarrow as pa

from math_rollouts.schema import (
    GROUP_KEY,
    NUCLEI_SCHEMA,
    POOL_SCHEMA,
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


def test_rollouts_carries_termination_and_lengths():
    names = set(ROLLOUTS_SCHEMA.names)
    # vLLM/OpenAI termination fields + the derived enum.
    assert {"finish_reason", "stop_reason", "terminal"} <= names
    # EOS-excluded lengths; the legacy num_tokens is renamed.
    assert {"prompt_num_tokens", "completion_num_tokens", "total_num_tokens"} <= names
    assert "num_tokens" not in names
    # raw rows do NOT carry the answer/match facts (those are pool/scores attributes).
    assert "answer_matches" not in names and "has_boxed" not in names


def test_pool_schema_is_rollouts_plus_criterion_free_facts():
    names = set(POOL_SCHEMA.names)
    assert set(ROLLOUTS_SCHEMA.names) <= names                # superset of raw rollouts
    assert {"answer_matches", "has_boxed", "answer_char_pos",
            "answer_token_frac", "dup_index"} <= names
    # no baked verdict / scorer on the pool — facts only.
    assert "is_correct" not in names and "scorer_id" not in names
    assert _dtype(POOL_SCHEMA, "answer_matches") == pa.bool_()
    assert _dtype(POOL_SCHEMA, "has_boxed") == pa.bool_()
    assert _dtype(POOL_SCHEMA, "dup_index") == pa.int32()


def test_single_split_aware_id():
    # The second id is gone; unique_id (split-aware) is the only problem identity.
    assert "math500_native_id" not in NUCLEI_SCHEMA.names
    assert "math500_native_id" not in ROLLOUTS_SCHEMA.names
    assert "unique_id" in NUCLEI_SCHEMA.names and "unique_id" in ROLLOUTS_SCHEMA.names


def test_scores_dtypes_and_join_keys():
    # the scored verdict is a 3-valued string (correct/incorrect/unresolved), not a
    # baked is_correct boolean; the criterion-free answer_matches fact rides along.
    assert "is_correct" not in SCORES_SCHEMA.names
    assert _dtype(SCORES_SCHEMA, "verdict") == pa.string()
    assert _dtype(SCORES_SCHEMA, "answer_matches") == pa.bool_()
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
