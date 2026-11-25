"""Export an NNUE .npz (bot/nnue.py format) to the plain files the Rust arena reads.

    python claude-playground/export_weights.py <in.npz> <out_dir>

Writes:
  net.txt  whitespace-separated: H, number of hidden layers n, then per hidden layer
           "n_out n_in residual(0/1)", then the output layer "n_out n_in" (6 n_in)
  net.bin  every array as little-endian float32, in this order: ft_w (1600 x H, row-major),
           ft_b (H), then per hidden layer w (n_out x n_in) and b (n_out), then out_w (6 x n_in)
           and out_b (6). Row-major, exactly as numpy stores them.

Layer keys are l1_w, l1_b, l2_w, ... (l{i}_w has shape (n_out, n_in)). The `residual` key is
optional (bot/nnue.py's loader defaults to no residual connections when it is missing).
"""

import sys
from pathlib import Path

import numpy as np

NUM_FEATURES = 1600
NUM_OUTPUTS = 6


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out_dir = Path(argv[0]), Path(argv[1])
    with np.load(src) as z:
        keys = set(z.files)
        arrs = {k: np.asarray(z[k]) for k in z.files}

    layers = sorted(int(k[1:-2]) for k in keys if k.startswith("l") and k.endswith("_w") and k[1:-2].isdigit())
    if layers != list(range(1, len(layers) + 1)):
        raise SystemExit(f"layer keys are not l1_w..l{len(layers)}_w: {layers}")
    n = len(layers)

    ft_w = arrs["ft_w"]
    ft_b = arrs["ft_b"]
    h = ft_w.shape[1]
    if ft_w.shape != (NUM_FEATURES, h) or ft_b.shape != (h,):
        raise SystemExit(f"ft_w {ft_w.shape} / ft_b {ft_b.shape} do not match ({NUM_FEATURES}, H) / (H,)")

    residual = arrs["residual"].astype(bool) if "residual" in arrs else np.zeros(n, dtype=bool)
    if residual.shape != (n,):
        raise SystemExit(f"residual has {residual.shape}, expected ({n},)")

    parts = [ft_w, ft_b]
    txt_layers = []
    width = 4 * h + 16
    for i in range(1, n + 1):
        w, b = arrs[f"l{i}_w"], arrs[f"l{i}_b"]
        if w.ndim != 2 or w.shape[1] != width or b.shape != (w.shape[0],):
            raise SystemExit(f"l{i}: weight {w.shape} / bias {b.shape} does not follow width {width}")
        if residual[i - 1] and w.shape[0] != w.shape[1]:
            raise SystemExit(f"l{i}: residual layer must be square, got {w.shape}")
        txt_layers.append(f"{w.shape[0]} {w.shape[1]} {int(residual[i - 1])}")
        parts += [w, b]
        width = w.shape[0]

    out_w, out_b = arrs["out_w"], arrs["out_b"]
    if out_w.shape != (NUM_OUTPUTS, width) or out_b.shape != (NUM_OUTPUTS,):
        raise SystemExit(f"out_w {out_w.shape} / out_b {out_b.shape} do not match ({NUM_OUTPUTS}, {width})")
    parts += [out_w, out_b]

    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [str(h), str(n)] + txt_layers + [f"{NUM_OUTPUTS} {width}"]
    (out_dir / "net.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    with open(out_dir / "net.bin", "wb") as f:
        for a in parts:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

    total = sum(p.size for p in parts)
    print(f"wrote {out_dir}: H={h}, {n} hidden layers, {total} float32 values "
          f"({4 * total / 1e6:.2f} MB), residual={residual.astype(int).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
