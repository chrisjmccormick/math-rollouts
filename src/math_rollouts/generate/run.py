"""CLI: build nuclei (HF) -> force K rollouts (vLLM) -> write RAW parquet + manifest.

Writes ``nuclei.parquet`` and ``rollouts.parquet`` (RAW, no correctness) plus a
``manifest.json`` describing the batch. Scoring is a SEPARATE pass
(``math-rollouts-score`` / ``score/run.py``) so raw rollouts stay pristine and
accuracy can be recomputed under any scorer.

  math-rollouts-generate --model Qwen/Qwen2.5-Math-1.5B --experiment math500_uniform_k16_d1 \
      --k 16 --max-depth 1 --out-root <local dataset root>

The output dir mirrors the HF dataset layout: ``<out-root>/generations/<slug>/<exp>/``.
GPU phases must run in the project env (``micromamba run -n guided-rollouts`` +
``source ~/env.sh``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from ..adapters import get_adapter
from ..config import GenConfig
from ..data.hf import experiment_dir
from ..schema import NUCLEI_SCHEMA, ROLLOUTS_SCHEMA, table_from_rows
from .rollouts import build_nuclei, force_rollouts


def _git_sha() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _load_problems(coverage: str, ids: list[str] | None):
    from ..data.problems import load_math500, load_math500_by_ids
    if ids:
        return load_math500_by_ids(ids)
    if coverage == "math500":
        return load_math500()
    raise ValueError(f"unknown coverage {coverage!r} (only 'math500' wired in v1; "
                     "pass --ids for a subset)")


def generate(*, model_id: str, experiment: str, out_root: str, k: int,
             max_depth: int, max_branch: int | None, run_id: int,
             seed: int | None, coverage: str, ids: list[str] | None,
             cfg: GenConfig | None = None, device: str = "cuda") -> Path:
    cfg = cfg or GenConfig()
    adapter = get_adapter(model_id)
    out_dir = Path(out_root) / experiment_dir(model_id, experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = _load_problems(coverage, ids)

    # ---- phase 1: nuclei (HF) ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device).eval()
    nuclei_rows = build_nuclei(model, tok, adapter, problems, cfg,
                               max_depth=max_depth, max_branch=max_branch, device=device)
    pq.write_table(table_from_rows(nuclei_rows, NUCLEI_SCHEMA),
                   out_dir / "nuclei.parquet")
    del model
    torch.cuda.empty_cache()

    # ---- phase 2: forced rollouts (vLLM) ----
    from vllm import LLM
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=cfg.max_model_len)
    rollout_rows = force_rollouts(llm, tok, adapter, nuclei_rows, problems, cfg,
                                  k=k, run_id=run_id, seed=seed)
    pq.write_table(table_from_rows(rollout_rows, ROLLOUTS_SCHEMA),
                   out_dir / "rollouts.parquet")

    manifest = {
        "model_id": model_id,
        "experiment": experiment,
        "is_thinking": bool(adapter.is_thinking),
        "gen_config": cfg.as_dict(),
        "gen_config_id": cfg.gen_config_id(),
        "k": k,
        "max_depth": max_depth,
        "max_branch": max_branch,
        "run_id": run_id,
        "seed": seed,
        "coverage": coverage,
        "n_problems": len(problems),
        "n_openers": len(nuclei_rows),
        "n_rollouts": len(rollout_rows),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[generate] wrote {out_dir}/nuclei.parquet, rollouts.parquet, manifest.json",
          flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build nuclei + force RAW rollouts.")
    ap.add_argument("--model", required=True, dest="model_id")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out-root", required=True,
                    help="local dataset root; output goes to generations/<slug>/<exp>/")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-branch", type=int, default=None)
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--coverage", default="math500", choices=["math500"])
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit native unique_ids (overrides --coverage)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    generate(model_id=a.model_id, experiment=a.experiment, out_root=a.out_root,
             k=a.k, max_depth=a.max_depth, max_branch=a.max_branch, run_id=a.run_id,
             seed=a.seed, coverage=a.coverage, ids=a.ids, device=a.device)


if __name__ == "__main__":
    main()
