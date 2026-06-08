"""Single source of truth for the dataset's parquet schemas + row builders.

Three tables, model-agnostic and forward-compatible with depth-N branch trees and
thinking models:

  nuclei.parquet    one row per OPENER (= per leaf of the nucleus tree).
  rollouts.parquet  RAW, no correctness — one row per forced sample.
  scores.parquet    DERIVED, re-runnable — one row per (rollout x scorer).

Grouping / statistics. The "these K were generated together" group key is
``(model_id, unique_id, branch_path, run_id)``; ``accuracy = sum(is_correct) /
group_size`` where ``group_size`` is the row count for that key (never stored as a
fragile count). ``run_id`` + ``gen_config_id`` + ``seed`` mark BATCH IDENTITY so
pooling across batches is deliberate, never accidental.

Depth-1 parity with the legacy ``openings_k16`` recipe: with ``max_depth=1``,
``branch_path == [i]`` and ``opener_token_ids[-1]`` reproduce the old ``token_id``.
``branch_path`` (not the raw fork token id) is the canonical opener identity, since
a token id can recur across different forks at depth>1.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa

# branch_path elements are small child-indices; opener/completion token ids are
# vocab ids (fit int32). path lists are short.
_I16 = pa.int16()
_I32 = pa.int32()

NUCLEI_SCHEMA = pa.schema([
    ("model_id", pa.string()),
    ("unique_id", pa.string()),                # <split>/<subj>/<n>; split in train|test|math500
    ("subject", pa.string()),
    ("answer", pa.string()),
    ("depth", pa.int8()),                      # leaf depth (1 for first-token nucleus)
    ("branch_path", pa.list_(_I16)),           # child-index at each fork, root->leaf
    ("opener_token_ids", pa.list_(_I32)),      # full forced prefix after the root
    ("opener_token_strs", pa.list_(pa.string())),
    ("fork_token_id", _I32),                   # branching token at this leaf's fork
    ("nuc_prob", pa.float32()),                # inbound (renormalized) prob of the fork choice
    ("path_prob", pa.float32()),               # product of inbound probs, root->leaf
    ("branch_size", _I16),                     # fork width at the leaf
    ("terminal", pa.string()),                 # eos / </think> / max-run / null
    ("is_thinking", pa.bool_()),
])

ROLLOUTS_SCHEMA = pa.schema([
    ("model_id", pa.string()),
    ("unique_id", pa.string()),
    ("subject", pa.string()),
    ("answer", pa.string()),
    ("depth", pa.int8()),
    ("branch_path", pa.list_(_I16)),
    ("opener_token_ids", pa.list_(_I32)),      # denormalized so rollouts stand alone
    ("run_id", _I32),                          # generation-batch id (batch identity)
    ("gen_config_id", _I32),
    ("seed", pa.int64()),                      # nullable
    ("temperature", pa.float32()),
    ("top_p", pa.float32()),
    ("max_gen_len", _I32),
    ("sample_idx", _I16),                      # 0..K-1 within the group
    ("completion_token_ids", pa.list_(_I32)),  # full response incl. forced opener
    ("completion_text", pa.string()),
    ("num_tokens", _I32),
    ("finish_reason", pa.string()),            # stop | length
])

SCORES_SCHEMA = pa.schema([
    ("model_id", pa.string()),
    ("unique_id", pa.string()),
    ("run_id", _I32),
    ("branch_path", pa.list_(_I16)),
    ("sample_idx", _I16),
    ("scorer_id", pa.string()),                # versioned scorer identity
    ("is_correct", pa.bool_()),
    ("answer_char_pos", _I32),                 # nullable
    ("answer_token_frac", pa.float32()),       # nullable
    ("leak_class", pa.string()),               # keeper | leak | unlocated | incorrect
])

# Join key from a score row back to its raw rollout.
ROLLOUT_KEY = ["model_id", "unique_id", "run_id", "branch_path", "sample_idx"]
# Generation-batch group ("these were sampled together"); accuracy denominator =
# row count over this key.
GROUP_KEY = ["model_id", "unique_id", "branch_path", "run_id"]


def table_from_rows(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a pyarrow Table from row dicts, coercing to ``schema`` (fills missing
    columns with null). Keeps every writer honest against one schema definition."""
    cols = {name: [r.get(name) for r in rows] for name in schema.names}
    return pa.table(cols, schema=schema)
