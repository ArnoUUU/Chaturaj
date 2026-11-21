"""Check that PyTorch sees the GPU, then time the model and a short search on it.

    python -m bot.gpu_check                 # report, works on CPU-only machines too
    python -m bot.gpu_check --require-gpu   # exit with an error if CUDA is missing (for Slurm jobs)

Writes the report to logs/gpu_check/<timestamp>_<host>.json as well as stdout.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import torch

from chaturaji.state import ChaturajiState

from . import registry
from .mcts import MCTS, Node


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_it(fn, device: str, repeat: int = 20) -> float:
    fn()  # warm-up
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeat


def run(device: str, batch: int, sims: int) -> dict:
    report = {"environment": registry.environment_info(device), "device": device, "checks": {}}
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 1),
                         "capability": f"{props.major}.{props.minor}"}
        a = torch.randn(4096, 4096, device=device)
        sec = time_it(lambda: a @ a, device, 10)
        report["checks"]["matmul_tflops_fp32"] = round(2 * 4096**3 / sec / 1e12, 2)

    state = ChaturajiState.initial()
    for kind in registry.KINDS:
        cfg = registry.default_config(kind)
        model = registry.build_model(kind, cfg).to(device)
        enc = registry.encoder(kind)(state)
        inputs = registry.collate(kind, [enc] * batch, device)
        model.eval()
        with torch.inference_mode():
            sec = time_it(lambda: model(*inputs), device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), 1e-3)

        def step():
            p, v, s = model(*inputs)
            loss = p.float().logsumexp(1).mean() + v.float().logsumexp(1).mean() + s.float().pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_sec = time_it(step, device, 5)
        model.eval()
        mcts = MCTS(registry.evaluator(kind, model, device), num_simulations=sims,
                    rng=np.random.default_rng(0))
        t0 = time.perf_counter()
        mcts.search(Node(state))
        search_sec = time.perf_counter() - t0
        report["checks"][kind] = {
            "params": registry.param_count(model),
            "inference_positions_per_sec": round(batch / sec),
            "train_step_ms": round(train_sec * 1000, 1),
            "mcts_sims_per_sec": round(sims / search_sec, 1),
        }
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--sims", type=int, default=64)
    args = ap.parse_args(argv)

    print(f"python {platform.python_version()} | torch {torch.__version__} | "
          f"CUDA available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
    if args.require_gpu and not torch.cuda.is_available():
        print("ERROR: no GPU visible to PyTorch. Check the job requested a GPU (--gpus / --gres),"
              " and that torch was installed with CUDA (torch.version.cuda is "
              f"{torch.version.cuda}).", file=sys.stderr)
        sys.exit(1)

    report = run(args.device, args.batch, args.sims)
    out = registry.LOGS_DIR / "gpu_check"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y-%m-%d_%H%M%S}_{platform.node() or 'host'}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))
    print(f"report saved to {path}")


if __name__ == "__main__":
    main()
