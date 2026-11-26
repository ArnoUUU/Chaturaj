"""Train the small NNUE on arena positions and export it for the Rust engine (CNN1 format).

Data: the position files `nnarena arena --out-pos` writes (one 152-byte record per position
before a move: board, mover, test seat, turn, game, scores, the mover's 16 globals, final
scores). Every position is a training example from the mover's view, greedy's moves
included: the label is who finished above whom (6 pairs, ties 0.5), the same 6 pairwise
logits the max^n search turns into each player's expected score.

Network (engine/src/nnue.rs): 1600 piece-square features per perspective -> h (the first
layer, small: default 32) for each of the 4 seats, clipped to [0, 1], mover first, + 16
globals -> 32 -> 32 -> 6 pairwise logits.

    python nn/train.py --data nn/runs/r0/pos.bin nn/runs/r1/pos.bin --out nn/runs/r1/net.bin
"""
import argparse, json, struct, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

NG_ALL = 36                 # 16 globals (Position::globals) + 20 extended (engine/src/features.rs)
REC = np.dtype([("board", np.uint8, 64), ("k", np.uint8), ("test_seat", np.uint8), ("turn", np.uint16),
                ("game", np.uint32), ("scores", np.int16, 4), ("glob", np.float32, NG_ALL), ("final", np.int16, 4)])
assert REC.itemsize == 232
NG = 36
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def feature_table():
    """FEAT[p, code, sq] = feature index for perspective p (1600 = empty, a zero row)."""
    t = np.full((4, 26, 64), 1600, dtype=np.int64)
    for p in range(4):
        for code in range(1, 26):
            owner, ty = (code - 1) // 5, (code - 1) % 5
            rel = 4 if owner == 4 else (owner - p) % 4
            for sq in range(64):
                r, c = sq // 8, sq % 8
                for _ in range(p):
                    r, c = 7 - c, r
                t[p, code, sq] = (rel * 5 + ty) * 64 + r * 8 + c
    return t


FEAT = feature_table()


def load(paths, game_offset_step=1_000_000):
    arrs = []
    for i, p in enumerate(paths):
        a = np.fromfile(p, dtype=REC)
        a = a.copy()
        a["game"] += i * game_offset_step       # games are unique per file
        arrs.append(a)
    return np.concatenate(arrs)


def future_points(a, gamma):
    """Per record: each seat's discounted points still to come in its game (absolute seats),
    F(t) = (s(t+1) - s(t)) + gamma * F(t+1), with s after the last record = final scores."""
    F = np.zeros((len(a), 4), np.float32)
    s = a["scores"].astype(np.float32)
    fin = a["final"].astype(np.float32)
    order = np.lexsort((a["turn"], a["game"]))
    g = a["game"][order]
    nxt = np.zeros(4, np.float32)
    for n in range(len(order) - 1, -1, -1):
        i = order[n]
        last = n == len(order) - 1 or g[n + 1] != g[n]
        if last:
            nxt = np.zeros(4, np.float32)
            inc = fin[i] - s[i]
        else:
            inc = s[order[n + 1]] - s[i]
        nxt = inc + gamma * nxt
        F[i] = nxt
    return F


def tensors(a, gamma=1.0):
    k = a["k"].astype(np.int64)
    persp = (k[:, None] + np.arange(4)[None, :]) % 4                       # (N, 4): mover, next, across, prev
    idx = FEAT[persp[:, :, None], a["board"][:, None, :].astype(np.int64), np.arange(64)[None, None, :]]  # (N,4,64)
    fin = a["final"].astype(np.float32)
    rel = np.take_along_axis(fin, persp, axis=1)                          # final scores, mover first
    y = np.stack([np.where(rel[:, i] > rel[:, j], 1.0, np.where(rel[:, i] == rel[:, j], 0.5, 0.0)) for i, j in PAIRS], 1)
    f = np.take_along_axis(future_points(a, gamma), persp, axis=1)
    return (torch.from_numpy(idx), torch.from_numpy(a["glob"][:, :NG].astype(np.float32)), torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(f))


class Net(nn.Module):
    """head "pairwise": the output layer gives the 6 logits. head "points": it gives each
    seat's expected future points F (mover first) and logit(i above j) =
    beta * ((s_i + F_i) - (s_j + F_j)) with the current scores s (globals 4..7 are s / 30)."""

    def __init__(self, h=32, hidden=(32, 32), head="points"):
        super().__init__()
        self.h, self.head = h, head
        self.ft = nn.Embedding(1601, h, padding_idx=1600)
        nn.init.normal_(self.ft.weight, 0, 0.05)
        with torch.no_grad():
            self.ft.weight[1600].zero_()
        self.ft_b = nn.Parameter(torch.full((h,), 0.25))
        layers, w = [], 4 * h + NG
        for n in hidden:
            l = nn.Linear(w, n)
            # clipped ReLU units die easily (all inputs outside (0, 1)): start them mid-range
            nn.init.normal_(l.weight, 0, 0.5 / w ** 0.5)
            nn.init.constant_(l.bias, 0.5)
            layers.append(l)
            w = n
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(w, 6 if head == "pairwise" else 4)
        if head == "points":
            nn.init.normal_(self.out.weight, 0, 0.01)
            nn.init.zeros_(self.out.bias)
            self.beta = nn.Parameter(torch.tensor(0.3))

    def raw(self, idx, g):
        acc = self.ft(idx).sum(2) + self.ft_b                              # (N, 4, h)
        x = torch.cat([acc.clamp(0, 1).flatten(1), g], 1)
        for l in self.layers:
            x = l(x).clamp(0, 1)
        return self.out(x)

    def logits(self, raw, g):
        if self.head == "pairwise":
            return raw
        v = 30.0 * g[:, 4:8] + raw
        return self.beta * torch.stack([v[:, i] - v[:, j] for i, j in PAIRS], 1)

    def forward(self, idx, g):
        return self.logits(self.raw(idx, g), g)

    def export(self, path):
        with open(path, "wb") as f:
            f.write(b"CNN1" + struct.pack("<II", self.h, len(self.layers)))
            f.write(self.ft.weight[:1600].detach().numpy().astype("<f4").tobytes())
            f.write(self.ft_b.detach().numpy().astype("<f4").tobytes())
            for l in list(self.layers) + [self.out]:
                f.write(struct.pack("<III", l.in_features, l.out_features, 0))
                f.write(l.weight.detach().numpy().astype("<f4").tobytes())
                f.write(l.bias.detach().numpy().astype("<f4").tobytes())
            if self.head == "points":
                f.write(struct.pack("<f", float(self.beta.detach())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.1, help="fraction of games held out")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--globals", type=int, choices=[16, 36], default=36, help="16 plain, 36 with the threat inputs")
    ap.add_argument("--head", choices=["points", "pairwise"], default="points")
    ap.add_argument("--gamma", type=float, default=1.0, help="per-ply discount of the future-points target")
    ap.add_argument("--aux", type=float, default=0.02, help="weight of the future-points MSE (points head)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    data = load(a.data)
    games = np.unique(data["game"])
    rng = np.random.default_rng(a.seed)
    val_games = set(rng.choice(games, max(1, int(len(games) * a.val)), replace=False).tolist())
    is_val = np.array([g in val_games for g in data["game"]])
    global NG
    NG = a.globals
    tr, va = tensors(data[~is_val], a.gamma), tensors(data[is_val], a.gamma)
    net = Net(a.h, tuple(a.hidden), a.head)
    if a.head == "points":
        with torch.no_grad():
            net.out.bias.copy_(tr[3].mean(0))       # start at the mean, so early steps don't kill units
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = max(1, (tr[0].shape[0] // a.batch))      # one epoch of linear warm-up
    step = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, log = 1e9, None, 0, []
    n = tr[0].shape[0]
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            b = perm[i:i + a.batch]
            raw = net.raw(tr[0][b], tr[1][b])
            loss = lossf(net.logits(raw, tr[1][b]), tr[2][b])
            if a.head == "points" and a.aux:
                loss = loss + a.aux * ((raw - tr[3][b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            step += 1
            if step <= warm:
                for gr in opt.param_groups:
                    gr["lr"] = a.lr * step / warm
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vb = vm = 0.0
            for i in range(0, va[0].shape[0], 4096):
                sl = slice(i, i + 4096)
                raw = net.raw(va[0][sl], va[1][sl])
                m = len(va[2][sl])
                vb += lossf(net.logits(raw, va[1][sl]), va[2][sl]).item() * m
                if a.head == "points":
                    vm += ((raw - va[3][sl]) ** 2).mean().item() * m
            vb, vm = vb / va[0].shape[0], vm / va[0].shape[0]
            vl = vb + (a.aux * vm if a.head == "points" else 0.0)
        log.append({"epoch": ep, "train": round(tot / n, 4), "val": round(vl, 4), "val_bce": round(vb, 4), "val_mse": round(vm, 3)})
        print(json.dumps(log[-1]), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    net.export(a.out)
    torch.save(net.state_dict(), Path(a.out).with_suffix(".pt"))
    info = {"positions": int(n), "val_positions": int(va[0].shape[0]), "games": int(len(games)), "best_val": round(best, 4),
            "h": a.h, "hidden": a.hidden, "head": a.head, "globals": a.globals, "gamma": a.gamma, "aux": a.aux, "wd": a.wd, "lr": a.lr, "params": sum(p.numel() for p in net.parameters()), "seconds": round(time.time() - t0, 1),
            "log": log, "data": a.data}
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "log"}))


if __name__ == "__main__":
    main()
