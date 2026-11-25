"""Device selection: registry.resolve_device and the scripts/gpu.py entry points (CPU-safe)."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bot import registry

ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / "scripts" / "gpu.py"
VENV = ROOT / ".venv-gpu"
CUDA = torch.cuda.is_available()


def gpu(*args, timeout=900):
    return subprocess.run([sys.executable, str(GPU), *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def test_auto_resolves_to_cpu_or_cuda():
    expected = "cuda" if CUDA else "cpu"
    assert registry.resolve_device("auto") == expected
    assert registry.resolve_device("cpu") == "cpu"


@pytest.mark.skipif(CUDA, reason="this machine has a CUDA GPU")
def test_cuda_without_a_gpu_exits_with_a_message():
    with pytest.raises(SystemExit, match="no CUDA GPU"):
        registry.resolve_device("cuda")


@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
def test_cuda_with_a_gpu_passes_through():
    assert registry.resolve_device("cuda") == "cuda"


def test_unknown_device_exits():
    with pytest.raises(SystemExit, match="unknown device"):
        registry.resolve_device("tpu")


def test_setup_dry_run_prints_the_cu126_install_and_creates_nothing():
    existed = VENV.exists()
    r = gpu("setup", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "download.pytorch.org/whl/cu126" in r.stdout
    assert VENV.exists() == existed


def test_check_exits_zero_without_require_gpu():
    r = gpu("check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "torch" in r.stdout


@pytest.mark.skipif(CUDA, reason="needs a machine without a GPU")
def test_check_require_gpu_exits_non_zero_without_a_gpu():
    r = gpu("check", "--require-gpu")
    assert r.returncode != 0
