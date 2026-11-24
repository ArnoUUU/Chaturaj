"""Model kinds, labelled saving to models/, and chunked logs in logs/.

Two model kinds share one interface:
  "mlp"    : encoding layer + 3 fully connected layers (bot/mlp.py), sparse inputs
  "resnet" : convolutional baseline (bot/model.py), dense 35x8x8 planes

Each saved model lives in its own folder, named by a label built from the run:

  models/<label>/model.pt     weights + optimizer + kind + config
  models/<label>/meta.json    run arguments, environment (GPU, versions, git commit),
                              per-iteration losses, evaluation results, status
  models/index.jsonl          one line per saved model (label, kind, metrics)

Opponent pool and archive (see models/README.md):
  models/opponent_pool/<name>.pt + .meta.json   league opponents; BEST_ in the name marks the best
  logs/models/<label>/                          retired model folders (weights, meta, snapshots)
find_model() turns any of these references into a weights file.

Logs for the same run go to logs/<label>/, split into numbered chunks.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
KINDS = ("mlp", "resnet")


# --- building, encoding, evaluating ---------------------------------------------------

def default_config(kind: str) -> dict:
    if kind == "mlp":
        return {"embed": 512, "widths": [1024, 512, 256], "dropout": 0.1}
    if kind == "resnet":
        return {"channels": 64, "blocks": 4, "value_hidden": 256}
    raise ValueError(f"unknown model kind {kind!r}")


def build_model(kind: str, config: dict) -> torch.nn.Module:
    if kind == "mlp":
        from .mlp import ChaturajiMLP
        return ChaturajiMLP(config["embed"], tuple(config["widths"]), config.get("dropout", 0.1))
    from .model import ChaturajiNet, NetConfig
    return ChaturajiNet(NetConfig(**config))


def encoder(kind: str):
    """state -> model input for one position (stored in training samples)."""
    if kind == "mlp":
        from .mlp import sparse_features
        return sparse_features
    from .features import encode
    return encode


def collate(kind: str, xs: list, device: str) -> tuple:
    """List of encoded positions -> positional args for model(*args)."""
    if kind == "mlp":
        from .mlp import collate_sparse
        return collate_sparse(xs, device)
    return (torch.from_numpy(np.stack(xs)).to(device),)


def evaluator(kind: str, model: torch.nn.Module, device: str):
    """MCTS evaluator for this kind of model."""
    if kind == "mlp":
        from .mlp import MLPEvaluator
        return MLPEvaluator(model, device)
    from .mcts import NetEvaluator
    return NetEvaluator(model, device)


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- labels and environment info ------------------------------------------------------

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip() or "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def size_tag(kind: str, config: dict) -> str:
    if kind == "mlp":
        return f"e{config['embed']}-w{'x'.join(str(w) for w in config['widths'])}"
    return f"c{config['channels']}-b{config['blocks']}"


def make_label(kind: str, config: dict, selection: str, sims: int, iterations: int,
               device: str, tag: str | None = None, when: datetime | None = None) -> str:
    """e.g. 2026-11-21_1530_smoke-cpu_mlp-e64-w128x64x32_ucb1-s16_it2_cpu
    (iterations=0 means a time-budgeted run and is labelled "timed")."""
    when = when or datetime.now()
    parts = [when.strftime("%Y-%m-%d_%H%M")]
    if tag:
        parts.append(tag)
    parts += [f"{kind}-{size_tag(kind, config)}", f"{selection}-s{sims}",
              f"it{iterations}" if iterations else "timed",
              "gpu" if str(device).startswith("cuda") else "cpu"]
    return "_".join(parts)


def environment_info(device: str) -> dict:
    info = {
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "device": str(device), "git_commit": git_commit(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for var in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CLUSTER_NAME", "CONDA_DEFAULT_ENV"):
        import os
        if os.environ.get(var):
            info[var.lower()] = os.environ[var]
    return info


# --- saving / loading -----------------------------------------------------------------

def model_dir(label: str, root: Path | None = None) -> Path:
    return (root or MODELS_DIR) / label


def label_for(path: str | Path) -> str:
    """The run label of a model file: models/<label>/model.pt, a pool file (its meta.json
    holds the label) or a snapshot (named by its file)."""
    p = Path(path)
    if p.name == "model.pt":
        return p.parent.name
    meta = p.with_name(p.stem + ".meta.json")
    if meta.is_file():
        return json.loads(meta.read_text(encoding="utf-8")).get("label") or p.stem
    return p.stem


def find_model(ref: str | Path) -> Path:
    """Resolve a model reference to its weights file. Accepts a .pt file (model.pt, a pool
    file or a snapshot), a folder, a label, or a pool name with or without BEST_. Searches
    models/, models/opponent_pool/, logs/models/ (retired models) and the repo root."""
    p = Path(ref)
    if p.is_file():
        return p
    if p.is_dir() and (p / "model.pt").is_file():
        return p / "model.pt"
    stem = p.parent.name if p.name == "model.pt" else (p.name[:-3] if p.name.endswith(".pt") else p.name)
    for base in (MODELS_DIR, MODELS_DIR / "opponent_pool", LOGS_DIR / "models", ROOT):
        for c in (base / p, base / stem / "model.pt", base / f"{stem}.pt", base / f"BEST_{stem}.pt"):
            if c.is_file():
                return c
    raise FileNotFoundError(f"no model for {str(ref)!r} in models/, models/opponent_pool/ or logs/models/")


def _checkpoint(model, kind: str, config: dict, label: str, meta: dict, optimizer=None) -> dict:
    return {"kind": kind, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "label": label, "step": meta.get("step", 0), "iteration": meta.get("iteration")}


def _atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a job killed mid-save never leaves a broken file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_model(model, kind: str, config: dict, label: str, meta: dict,
               optimizer=None, root: Path | None = None) -> Path:
    folder = model_dir(label, root)
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_save(_checkpoint(model, kind, config, label, meta, optimizer), folder / "model.pt")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return folder / "model.pt"


def save_snapshot(model, kind: str, config: dict, label: str, meta: dict, name: str,
                  optimizer=None, root: Path | None = None) -> Path:
    """Keep a copy as models/<label>/snapshots/<name>.pt (loadable with load_model / --resume)."""
    folder = model_dir(label, root) / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pt"
    _atomic_save(_checkpoint(model, kind, config, label, meta, optimizer), path)
    return path


def register(label: str, meta: dict, root: Path | None = None) -> None:
    """Append a finished model to models/index.jsonl."""
    root = root or MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    entry = {"label": label, "kind": meta.get("kind"), "params": meta.get("params"),
             "status": meta.get("status"), "evaluation": meta.get("evaluation"),
             "final_loss": (meta.get("history") or [{}])[-1].get("loss"),
             "device": meta.get("environment", {}).get("device"),
             "git_commit": meta.get("environment", {}).get("git_commit"),
             "saved_at": datetime.now().isoformat(timespec="seconds")}
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_model(path: str | Path, device: str = "cpu"):
    """Load models/<label>/model.pt (or the folder). Returns (model, kind, config, checkpoint)."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["kind"], ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["kind"], ckpt["config"], ckpt


# --- chunked logs ---------------------------------------------------------------------

class ChunkedLogger:
    """JSON-lines logger that starts a new file every `chunk_lines` records:
    logs/<label>/log_0000.jsonl, log_0001.jsonl, ... Also mirrors messages to stdout."""

    def __init__(self, label: str, chunk_lines: int = 200, root: Path | None = None,
                 echo: bool = True):
        self.dir = (root or LOGS_DIR) / label
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_lines = chunk_lines
        self.echo = echo
        existing = sorted(self.dir.glob("log_*.jsonl"))
        self.chunk = int(existing[-1].stem.split("_")[1]) if existing else 0
        self.lines = (sum(1 for _ in existing[-1].open(encoding="utf-8")) if existing else 0)

    @property
    def path(self) -> Path:
        return self.dir / f"log_{self.chunk:04d}.jsonl"

    def log(self, event: str, **fields) -> None:
        if self.lines >= self.chunk_lines:
            self.chunk += 1
            self.lines = 0
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self.lines += 1
        if self.echo:
            print(json.dumps(record, default=str), flush=True)
