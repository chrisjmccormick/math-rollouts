"""Model registry — the ONLY place model ids are enumerated.

``get_adapter(model_id)`` returns the right ModelAdapter. Add a model by adding a
line here (and a subclass); the generator, schema, and scoring are untouched.
"""
from __future__ import annotations

from .base import ModelAdapter
from .deepseek_distill import DeepseekR1DistillAdapter
from .paper_base import PaperBaseAdapter
from .qwen3_think import Qwen3ThinkAdapter
from .qwen_math import QwenMathAdapter

# Explicit, exact-match registrations.
_EXACT = {
    "Qwen/Qwen2.5-Math-1.5B": lambda mid: QwenMathAdapter(mid),
    "sail/Qwen2.5-Math-1.5B-Oat-Zero": lambda mid: QwenMathAdapter(mid),
    "Qwen/Qwen3-8B": lambda mid: Qwen3ThinkAdapter(mid),
    "Qwen/Qwen3-8B-Base": lambda mid: PaperBaseAdapter(mid),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": lambda mid: DeepseekR1DistillAdapter(mid),
}


def get_adapter(model_id: str, *, family: str | None = None) -> ModelAdapter:
    """Return the adapter for ``model_id``. ``family`` ('qwen_math' | 'qwen3_think'
    | 'paper_base' | 'deepseek_distill') forces a choice for unregistered ids."""
    if family is not None:
        return {
            "qwen_math": QwenMathAdapter,
            "qwen3_think": Qwen3ThinkAdapter,
            "paper_base": PaperBaseAdapter,
            "deepseek_distill": DeepseekR1DistillAdapter,
        }[family](model_id)
    if model_id in _EXACT:
        return _EXACT[model_id](model_id)
    raise KeyError(f"no adapter registered for {model_id!r}; known: {sorted(_EXACT)} "
                   f"(or pass family=...)")
