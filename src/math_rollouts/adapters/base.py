"""ModelAdapter — the single abstraction over thinking vs non-thinking models.

Everything the unified generator needs to differ on between model families lives
here: how the prompt is built up to the nucleus ROOT, which tokens TERMINATE a
branch, how a completion is SCORED, and the vLLM stop strings. The nucleus tree,
the forced-rollout step, and the parquet schema are all model-agnostic and never
branch on model family — they call through the adapter.

Adding a model later = one new subclass + one registry line. No schema change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    model_id: str
    is_thinking: bool

    @abstractmethod
    def prompt_ids(self, problem: dict, tok) -> list[int]:
        """Token ids up to (and including) the nucleus ROOT position.

        Non-thinking: the chat/template prefix ending at the assistant turn, so the
        nucleus is over the first OUTPUT token. Thinking: additionally forces
        ``<think>\\n`` so the nucleus is over the first REASONING token."""

    @abstractmethod
    def terminal_ids(self, tok) -> dict[int, str]:
        """Map token_id -> terminal reason. Always includes EOS; thinking models add
        ``</think>`` so branch expansion stops at the end of reasoning."""

    @abstractmethod
    def score(self, completion_text: str, answer: str) -> bool:
        """Default correctness for this model family (non-thinking: full text;
        thinking: post-``</think>`` only). The standalone scoring pass can override
        with any registered scorer; this is the family default."""

    def vllm_stop(self) -> list[str]:
        """Stop strings for vLLM forced rollouts (e.g. the chat end token)."""
        return []
