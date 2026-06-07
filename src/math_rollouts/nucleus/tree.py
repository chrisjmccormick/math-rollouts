"""Unified nucleus/branch tree (anytree-backed), the single code path for both
thinking and non-thinking models.

A node represents ONE chosen fork token. The (virtual) root sits at the prompt
position supplied by the model adapter; its nucleus is the first-token nucleus, and
its children are the nucleus members (depth 1). Expanding a child advances one
token and forks again, to ``max_depth``. ``max_depth=1`` therefore reproduces the
classic first-token nucleus exactly: one single-token opener per nucleus member —
byte-parity with the legacy ``openings_k16`` recipe.

Efficiency: ONE persistent ``DynamicCache`` walked DFS with single-token forwards;
``cache.crop()`` on backtrack so each tree token is forwarded once. Device-agnostic.

The nucleus is the model adapter's notion of the root position (e.g. the first
reasoning token after a forced ``<think>\\n`` for thinking models); terminal tokens
(EOS, ``</think>``) come from the adapter, so a terminal fork member becomes a
terminal leaf with no expansion. Nothing here branches on model family.
"""
from __future__ import annotations

import torch
from anytree import Node
from transformers.cache_utils import DynamicCache


class NucleusTree:
    def __init__(self, model, tok, adapter, cfg, *, max_depth: int = 1,
                 max_branch: int | None = None, node_budget: int = 100_000,
                 device: str = "cuda"):
        self.model, self.tok, self.adapter, self.cfg = model, tok, adapter, cfg
        self.max_depth = max_depth
        self.max_branch = max_branch if max_branch is not None else cfg.top_k
        self.node_budget = node_budget
        self.device = device
        self.terminals = adapter.terminal_ids(tok)   # token_id -> reason
        self.cache = None
        self._nodes = 0

    # ---- model / nucleus ----
    def _nucleus(self, logits):
        """Renormalized nucleus: softmax(logits/T), top_k cap, top_p keep (always
        keep the top token). Returns (ids, probs) parallel lists."""
        pr = torch.softmax(logits.float() / self.cfg.temperature, dim=-1)
        sp, si = torch.sort(pr, descending=True)
        sp, si = sp[: self.cfg.top_k], si[: self.cfg.top_k]
        keep = (torch.cumsum(sp, 0) - sp) < self.cfg.top_p
        keep[0] = True
        ids = [int(t) for t in si[keep].tolist()]
        p = sp[keep]
        p = (p / p.sum()).tolist()
        return ids, [round(x, 6) for x in p]

    def _prime(self, ids):
        self.cache = DynamicCache()
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device),
                             past_key_values=self.cache, use_cache=True)
        return out.logits[0, -1]

    def _advance(self, tok_id):
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([[tok_id]], device=self.device),
                             past_key_values=self.cache, use_cache=True)
        return out.logits[0, -1]

    # ---- walk ----
    def _new_node(self, parent, token_id, child_idx, inbound_prob, branch_size, depth):
        self._nodes += 1
        path = (parent.branch_path + [child_idx]) if parent is not None else []
        term = self.terminals.get(token_id)
        node = Node(
            f"d{depth} i{child_idx} p={inbound_prob:.3f}"
            f"{'' if not term else f' [{term}]'} {self.tok.decode([token_id])!r}",
            parent=parent, token_id=token_id, branch_path=path, depth_level=depth,
            inbound_prob=inbound_prob,
            path_prob=(parent.path_prob if parent is not None else 1.0) * inbound_prob,
            branch_size=branch_size, terminal=term, nuc_ids=[], nuc_probs=[],
        )
        return node

    def _expand(self, node, logits, depth):
        """Fork ``node`` (whose logits are known) into its nucleus children, recurse
        to max_depth. Cache is positioned at ``node`` on entry and restored on exit."""
        ids, probs = self._nucleus(logits)
        node.nuc_ids, node.nuc_probs = ids, probs
        n_keep = min(len(ids), self.max_branch)
        seq_len = self.cache.get_seq_length()
        for child_idx, (cid, cprob) in enumerate(list(zip(ids, probs))[:n_keep]):
            if self._nodes >= self.node_budget:
                return
            child = self._new_node(node, cid, child_idx, cprob, branch_size=len(ids), depth=depth)
            if child.terminal is None and depth < self.max_depth:
                clogits = self._advance(cid)
                self._expand(child, clogits, depth + 1)
                self.cache.crop(seq_len)   # backtrack to the fork position
        return node

    def build(self, prompt_ids):
        """Build the tree from the adapter's prompt ids; returns the virtual root
        (depth 0). Leaves are the openers (see ``leaf_openers``)."""
        self._nodes = 0
        logits = self._prime(prompt_ids)
        root = Node("root", parent=None, token_id=None, branch_path=[], depth_level=0,
                    inbound_prob=1.0, path_prob=1.0, branch_size=0, terminal=None,
                    nuc_ids=[], nuc_probs=[])
        self._expand(root, logits, depth=1)
        return root
