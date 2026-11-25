"""GPU setup and runs for Chaturaji: CUDA PyTorch in .venv-gpu, with a CPU fallback everywhere.

Run it from the repo root of a fresh clone. cluster/environment.yml pins Python 3.12, and
PyTorch's CUDA wheels must exist for the Python you use, so prefer 3.12 or 3.13.

Windows (PowerShell):
    git clone <repo-url> Chaturaj
    cd Chaturaj
    py -3.12 scripts/gpu.py setup
    python scripts/gpu.py check --require-gpu
    python scripts/gpu.py selfplay --hours 8 --depth 5 --tag gtx1660

Linux:
    git clone <repo-url> Chaturaj
    cd Chaturaj
    python3 scripts/gpu.py setup
    python3 scripts/gpu.py check --require-gpu
    python3 scripts/gpu.py selfplay --hours 8 --depth 5 --tag gtx1660

Commands:
    setup [--cuda 126|128|130] [--dry-run]
        Create .venv-gpu (if missing), install CUDA PyTorch from download.pytorch.org
        (default cu126: supports the GTX 1660's sm_75), then numpy, matplotlib, pandas,
        pytest, selenium and maturin, then build the Rust engine (engine/) and install its
        wheel. Needs cargo for the engine (https://rustup.rs); without it the engine step is
        skipped. Ends with `check`. --dry-run prints the commands and creates nothing.
    check [--require-gpu]
        Print the Python, torch and CUDA versions and the GPU (name, compute capability,
        memory); warn on a compute capability below 7.5 or a PyTorch build without sm_75; run
        a matmul on the GPU, then `python -m bot.gpu_check`. --require-gpu exits non-zero
        when there is no GPU.
    train-nnue | selfplay | evaluate | benchmark [--device auto|cpu|cuda] [...]
        Run `python -m bot.<module> --device <dev> ...` with .venv-gpu's Python when it
        exists, else the current Python. --device defaults to auto; every other argument is
        passed through unchanged.
    test [-k EXPR]
        pytest on tests/ (or only tests matching -k EXPR), then check.

What runs where: the max^n self-play search runs on the CPU, in the compiled Rust engine.
The GPU speeds up the NNUE training steps (train-nnue, and the training inside selfplay)
and the network evaluation in MCTS (evaluate, benchmark).

Batch size on the GTX 1660 (6 GB): the NNUE presets are small enough that --batch (default
1024) is not the limit for small, medium or large. For --preset xl (63.6 M weights, 255 MB
of float32, about 1 GB with Adam state and gradients by a parameter-count estimate, not
measured on the 1660), keep --batch at 1024 and use --amp to halve activation memory; if
CUDA reports out-of-memory, halve --batch (512, then 256). The self-play --buffer lives in
CPU RAM, not on the GPU, so it does not count against the 6 GB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv-gpu"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu{}"
DEPS = ["numpy", "matplotlib", "pandas", "pytest", "selenium", "maturin"]
MODULES = {"train-nnue": "train_nnue", "selfplay": "selfplay_nnue", "evaluate": "evaluate",
           "benchmark": "benchmark"}


def venv_python() -> Path:
    return VENV / "Scripts" / "python.exe" if os.name == "nt" else VENV / "bin" / "python"


def python_for_bot() -> str:
    """The venv's Python when .venv-gpu exists, else the one running this script."""
    p = venv_python()
    return str(p) if p.exists() else sys.executable


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def run(cmd, cwd: Path = ROOT, dry: bool = False) -> int:
    """Print the command, then run it (unless dry)."""
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd, cwd=str(cwd))


def cmd_setup(args) -> int:
    dry = args.dry_run
    if venv_python().exists():
        print(f"using the existing {VENV}", flush=True)
    elif run([sys.executable, "-m", "venv", VENV], dry=dry):
        return 1
    py = str(venv_python())
    torch_cmd = [py, "-m", "pip", "install", "--upgrade", "torch",
                 "--index-url", PYTORCH_INDEX.format(args.cuda)]
    steps = [
        [py, "-m", "pip", "install", "--upgrade", "pip"],
        torch_cmd,
        [py, "-m", "pip", "install", *DEPS],
    ]
    for step in steps:
        if run(step, dry=dry):
            return 1

    if shutil.which("cargo") is None:
        print("cargo (Rust) is not installed, so the compiled engine is skipped. The NNUE search then "
              "falls back to NumPy and is much slower. Install Rust from https://rustup.rs (default "
              "options), open a new terminal, and rerun `python scripts/gpu.py setup`.", flush=True)
    else:
        engine = ROOT / "engine"
        if run([py, "-m", "maturin", "build", "--release", "-i", py], cwd=engine, dry=dry):
            return 1
        wheels_dir = engine / "target" / "wheels"
        if dry:
            print(f"+ {py} -m pip install --force-reinstall --no-deps <newest {wheels_dir}/chaturaji_engine-*.whl>")
        else:
            wheels = sorted(wheels_dir.glob("chaturaji_engine-*.whl"), key=lambda p: p.stat().st_mtime)
            if not wheels:
                print(f"no engine wheel found in {wheels_dir}", file=sys.stderr)
                return 1
            if run([py, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheels[-1]]):
                return 1

    print("\nchecking the setup:", flush=True)
    return run([py, Path(__file__).resolve(), "check"], dry=dry)


def cmd_check(args) -> int:
    import torch

    print(f"python  {sys.version.split()[0]}  {sys.executable}")
    print(f"torch   {torch.__version__}  (built for CUDA: {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("CUDA    not available (no NVIDIA GPU or driver, or a CPU-only PyTorch build)")
        if args.require_gpu:
            print("error: --require-gpu, but PyTorch sees no CUDA device", file=sys.stderr)
            return 1
        print("running on the CPU")
    else:
        n = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        cap = (p.major, p.minor)
        print(f"GPU     {p.name}  compute capability {cap[0]}.{cap[1]}  "
              f"memory {p.total_memory / 2**30:.1f} GB" + (f"  ({n} GPUs, device 0 used)" if n > 1 else ""))
        if cap < (7, 5):
            print("WARNING: compute capability below 7.5 (the GTX 1660 is 7.5); some kernels may be missing")
        archs = torch.cuda.get_arch_list()
        if "sm_75" not in archs:
            print(f"WARNING: this PyTorch build has no sm_75 kernels (arch list: {' '.join(archs)}); "
                  "reinstall with `python scripts/gpu.py setup --cuda 126`")
        try:
            a = torch.randn(512, 512, device="cuda")
            checksum = float((a @ a).sum().item())
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"error: a matmul on the GPU failed: {e}", file=sys.stderr)
            return 1
        print(f"matmul  512x512 on the GPU ok (checksum {checksum:.1f})")

    cmd = [sys.executable, "-m", "bot.gpu_check"]
    if args.require_gpu:
        cmd.append("--require-gpu")
    if run(cmd):
        # Reported, not fatal: bot/gpu_check.py's timing loop unpacks three outputs for every
        # model kind, which the NNUE does not produce. The GPU checks above decide the status.
        print("WARNING: python -m bot.gpu_check exited non-zero (see the traceback above); "
              "the matmul and device checks above still stand", flush=True)
    return 0


def cmd_bot(args, extra) -> int:
    cmd = [python_for_bot(), "-m", f"bot.{MODULES[args.cmd]}", "--device", args.device, *extra]
    return run(cmd)


def cmd_test(args) -> int:
    py = python_for_bot()
    cmd = [py, "-m", "pytest", "-q", "tests"]
    if args.k:
        cmd += ["-k", args.k]
    rc = run(cmd)
    crc = run([py, Path(__file__).resolve(), "check"])
    return rc or crc


def main(argv=None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="create .venv-gpu with CUDA PyTorch, the dependencies and the engine")
    s.add_argument("--cuda", choices=["126", "128", "130"], default="126", help="CUDA wheel index (default 126)")
    s.add_argument("--dry-run", action="store_true", help="print the commands only")
    c = sub.add_parser("check", help="report the GPU and run a small GPU test")
    c.add_argument("--require-gpu", action="store_true", help="exit non-zero if there is no GPU")
    for name, module in MODULES.items():
        b = sub.add_parser(name, help=f"run bot.{module} (other arguments are passed through)")
        b.add_argument("--device", default="auto", help="auto (default), cpu or cuda")
    t = sub.add_parser("test", help="pytest, then check")
    t.add_argument("-k", default=None, metavar="EXPR", help="run only the tests matching EXPR")
    args, extra = ap.parse_known_args(raw)
    if extra and args.cmd not in MODULES:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd == "setup":
        return cmd_setup(args)
    if args.cmd == "check":
        # torch must come from the venv (CUDA build), so re-run there when it exists
        if venv_python().exists() and not in_venv():
            return subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *raw])
        return cmd_check(args)
    if args.cmd == "test":
        return cmd_test(args)
    return cmd_bot(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
