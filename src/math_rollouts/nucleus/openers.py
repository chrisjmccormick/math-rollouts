"""Turn a built ``NucleusTree`` into per-opener rows (the durable opener identity
carried into rollouts).

A leaf of the tree is one OPENER: the forced token sequence from the virtual root
down to (and including) that leaf. ``leaf_openers(root)`` walks the tree and yields a
dict per leaf with the tree-derived fields of ``NUCLEI_SCHEMA`` — the model/problem
fields (``model_id``, ``unique_id``, ...) are filled in by the caller that knows the
problem.

``branch_path`` (child-index at each fork, root->leaf) is the canonical opener
identity: depth-safe, since a raw fork token id can recur across different forks at
depth>1. For the depth-1 first-token nucleus, ``branch_path == [i]`` and
``opener_token_ids == [fork_token_id]`` — byte-parity with the legacy ``openings_k16``
``token_id``.
"""
from __future__ import annotations

from anytree import Node


def _is_leaf(node: Node) -> bool:
    return not node.children


def leaf_openers(root: Node, tok) -> list[dict]:
    """Walk ``root`` and return one opener dict per leaf (DFS, deterministic order).

    Each dict carries the tree-side ``NUCLEI_SCHEMA`` fields::

        depth, branch_path, opener_token_ids, opener_token_strs,
        fork_token_id, nuc_prob, path_prob, branch_size, terminal

    The opener prefix is the chain of chosen fork tokens from the first child below
    the root down to the leaf (the root itself sits at the prompt position and
    contributes no token)."""
    openers: list[dict] = []
    for node in _iter_leaves(root):
        chain = list(node.path)[1:]            # drop the virtual root
        token_ids = [n.token_id for n in chain]
        openers.append(dict(
            depth=int(node.depth_level),
            branch_path=list(node.branch_path),
            opener_token_ids=token_ids,
            opener_token_strs=[tok.decode([t]) for t in token_ids],
            fork_token_id=int(node.token_id),
            nuc_prob=float(node.inbound_prob),
            path_prob=float(node.path_prob),
            branch_size=int(node.branch_size),
            terminal=node.terminal,
        ))
    return openers


def _iter_leaves(root: Node):
    """Yield leaves in DFS (child-index) order — matches expansion order so opener
    rows line up with how the tree was built."""
    if _is_leaf(root) and root.parent is not None:
        yield root
        return
    for child in root.children:
        yield from _iter_leaves(child)
