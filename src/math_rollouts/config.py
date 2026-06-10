"""Generation config + environment guards.

Importing this module sets ``VLLM_USE_FLASHINFER_SAMPLER=0`` BEFORE any vLLM
import. FlashInfer's top-p/top-k sampler JIT-compiles and needs ``nvcc``/CUDA_HOME,
which isn't present on every box; disabling it keeps vLLM engine init from
crashing. (Mirrors the source project's ``run_random_nothink.py`` recipe.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

# Must be set before `import vllm` anywhere downstream. setdefault so an explicit
# override in the environment still wins.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Under a Jupyter/Colab kernel, run the V1 engine IN-PROCESS: the engine-core
# subprocess dies opaquely there ("Failed core proc(s): {}" with the root cause
# swallowed), and the in-process path lets generate.natural's stdio guard handle
# ipykernel's fileno-less streams (vLLM init calls sys.stdout.fileno()).
if "ipykernel" in sys.modules:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


@dataclass(frozen=True)
class GenConfig:
    """Sampling + nucleus config, fixed across the project's canonical runs.

    Rollouts are sampled with ``temperature`` + ``top_p``, plus any per-family
    ``adapter.sampling_overrides()`` (e.g. Qwen3's vendor thinking-mode
    ``top_k=20``). This config's ``top_k`` is UNRELATED to sampling — it caps the
    **nucleus** fan-out (the first-token / branch set, computed on
    temperature-scaled probs, capped at ``top_k``) and the size of the post-hoc
    per-token nucleus store (``analysis.token_nuclei``).
    """

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20          # nucleus-size cap ONLY (not a sampling limiter)
    max_tokens: int = 3000   # max generated tokens per rollout
    max_model_len: int = 4096

    def gen_config_id(self) -> int:
        """Stable id for this sampling config (200 == the project's RANDOM config).

        Carried on every rollout row so pooling across batches is deliberate: rows
        with different ``gen_config_id`` are NOT the same sampling distribution.
        """
        if (self.temperature, self.top_p, self.top_k, self.max_tokens) == (0.6, 0.95, 20, 3000):
            return 200
        # Non-canonical configs get a deterministic-ish hash id in a separate band.
        return 900 + (hash((self.temperature, self.top_p, self.top_k, self.max_tokens)) % 99)

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = GenConfig()
