"""NNUE value network for a max^n alpha-beta engine (see logs/nnue_alphabeta_plan.md).

    board --4 perspectives--> 4 accumulators (1600 -> H each, shared weights, updated per move)
          --clip [0,1], mover's first, + 16 globals--> hidden layers --> 6 pairwise logits

Each perspective is one seat's view of the board: the same 1,600 (relative owner, piece,
square) features the MLP uses, rotated to that seat. One weight matrix serves all four
perspectives, because the features are already relative to the viewer.

Hidden layers are fully connected with clipped ReLU. With `residual`, a layer whose width
equals the previous one also adds its input back (h + crelu(W h + b)), so the deep presets
(7 layers) train like a ResNet. PRESETS lists the sizes: "small" is the original 2 x 32
network (1.8 MB), "xl" is 255 MB of float32 weights.

The output is the 6 pairwise logits between the four seats, in mover-relative order
(0 = mover, 1 = next, 2 = across, 3 = previous): logit[i, j] for the pairs in PAIRS is
P(seat i finishes above seat j), with a tie counting as half. Each seat's expected
chess.com FFA score is the mean of its three pairwise probabilities, which gives max^n
the utility vector it needs (one value per player) from a single evaluation.

Implementations of the same network:
  NNUE           PyTorch, for training (batched, full recompute)
  NumpyNNUE /    NumPy reference for search: accumulators updated incrementally from the
  Accumulator    board difference of a move
  engine/        the compiled Rust core (Python module `chaturaji_engine`). It must match
                 the NumPy reference, and is used automatically when it is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chaturaji.constants import OWNERS, PIECE_TYPES
from chaturaji.state import Board, ChaturajiState

from .features import mover_index, rotate_square, to_absolute
from .mlp import NUM_GLOBALS, NUM_PIECE_FEATURES, global_features

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_OWNER_CODE = {o: i for i, o in enumerate(OWNERS)}           # R B Y G X -> 0..4
_TYPE_IDX = {t: i for i, t in enumerate(PIECE_TYPES)}

PRESETS = {
    # accumulator width per perspective, hidden layer widths, residual connections
    "small": {"acc": 256, "hidden": [32, 32], "residual": False},       # 0.44 M weights, 1.8 MB
    "medium": {"acc": 512, "hidden": [512] * 7, "residual": True},      # 3.5 M, 14 MB
    "large": {"acc": 1024, "hidden": [1024] * 7, "residual": True},     # 12.2 M, 49 MB
    "xl": {"acc": 2048, "hidden": [2560] * 7, "residual": True},        # 63.6 M, 255 MB
}


def _build_feature_table() -> np.ndarray:
    """FEATURE[p, owner, type, r, c] = feature index of that piece seen from seat p."""
    t = np.zeros((4, 5, len(PIECE_TYPES), 8, 8), dtype=np.int64)
    for p in range(4):
        for o in range(5):
            rel = 4 if o == 4 else (o - p) % 4
            for ty in range(len(PIECE_TYPES)):
                for r in range(8):
                    for c in range(8):
                        cr, cc = rotate_square(r, c, p)
                        t[p, o, ty, r, c] = (rel * 5 + ty) * 64 + cr * 8 + cc
    return t


FEATURE = _build_feature_table()
_REL_ORDER = [[(k + i) % 4 for i in range(4)] for k in range(4)]
_TO_ABS = [np.roll(np.arange(4), k) for k in range(4)]   # relative -> absolute seat order


def piece_features(piece: str, r: int, c: int) -> np.ndarray:
    """Feature index of one piece, from each of the four seats (absolute R, B, Y, G)."""
    return FEATURE[:, _OWNER_CODE[piece[0]], _TYPE_IDX[piece[1]], r, c]


def perspective_indices(board: Board) -> list[np.ndarray]:
    """Active feature indices of the whole board, one array per absolute seat."""
    rows = [piece_features(p, r, c) for r, row in enumerate(board)
            for c, p in enumerate(row) if p is not None]
    if not rows:
        return [np.zeros(0, dtype=np.int64) for _ in range(4)]
    a = np.stack(rows, axis=1)                                # (4, pieces)
    return [a[p] for p in range(4)]


def board_diff(old: Board, new: Board) -> tuple[list[tuple], list[tuple]]:
    """Pieces removed and added between two boards, as (piece, r, c)."""
    removed, added = [], []
    for r in range(8):
        for c in range(8):
            a, b = old[r][c], new[r][c]
            if a != b:
                if a is not None:
                    removed.append((a, r, c))
                if b is not None:
                    added.append((b, r, c))
    return removed, added


def board_codes(board: Board) -> np.ndarray:
    """64 bytes for the compiled core: 0 = empty, else 1 + owner (R B Y G X = 0..4) * 5 + type."""
    return np.fromiter((0 if p is None else 1 + _OWNER_CODE[p[0]] * 5 + _TYPE_IDX[p[1]]
                        for row in board for p in row), dtype=np.uint8, count=64)


def pairwise_targets(score_rel: np.ndarray) -> np.ndarray:
    """6 targets from final scores (or margins) in mover-relative order: 1 / 0.5 / 0."""
    return np.array([0.5 if score_rel[i] == score_rel[j] else float(score_rel[i] > score_rel[j])
                     for i, j in PAIRS], dtype=np.float32)


_U_MAT = np.zeros((4, len(PAIRS)))
_U_OFF = np.zeros(4)
for _n, (_i, _j) in enumerate(PAIRS):
    _U_MAT[_i, _n] += 1.0          # seat i gets p
    _U_MAT[_j, _n] -= 1.0          # seat j gets 1 - p
    _U_OFF[_j] += 1.0


def utilities(p_pairs: np.ndarray) -> np.ndarray:
    """6 pairwise probabilities (…, 6) -> each seat's expected FFA score (…, 4), relative."""
    return (np.asarray(p_pairs, dtype=np.float64) @ _U_MAT.T + _U_OFF) / 3.0


# --- PyTorch model (training) -----------------------------------------------------------

class NNUE(nn.Module):
    def __init__(self, acc: int = 256, hidden: tuple[int, ...] = (32, 32), residual: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.acc_size = acc
        self.ft = nn.EmbeddingBag(NUM_PIECE_FEATURES, acc, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(acc))
        widths = [4 * acc + NUM_GLOBALS, *hidden]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.skip = [bool(residual and a == b) for a, b in zip(widths[:-1], widths[1:])]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(widths[-1], len(PAIRS))
        nn.init.normal_(self.ft.weight, 0.0, 0.05)
        with torch.no_grad():
            self.ft_bias.fill_(0.25)
            for layer, skip in zip(self.layers, self.skip):
                if skip:                              # residual branches start small
                    layer.weight.mul_(0.5)

    def body(self, acc: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        """acc: (B, 4, H) pre-activation accumulators in mover-relative order."""
        h = torch.clamp(acc, 0.0, 1.0).reshape(acc.shape[0], -1)
        h = torch.cat([h, globals_], 1)
        for layer, skip in zip(self.layers, self.skip):
            y = torch.clamp(layer(h), 0.0, 1.0)
            h = self.drop(y + h if skip else y)
        return self.out(h)

    def forward(self, indices, offsets, globals_):
        """Bags are ordered (position 0: mover, next, across, prev; position 1: …)."""
        acc = self.ft(indices, offsets) + self.ft_bias
        return self.body(acc.view(globals_.shape[0], 4, self.acc_size), globals_)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also accepts the first checkpoints' names (l1, l2 instead of layers.0, layers.1)."""
        renamed = {}
        for k, v in state_dict.items():
            if len(k) > 1 and k[0] == "l" and k[1].isdigit():
                n, rest = k[1:].split(".", 1)
                k = f"layers.{int(n) - 1}.{rest}"
            renamed[k] = v
        return super().load_state_dict(renamed, strict=strict, assign=assign)


def nnue_features(state: ChaturajiState) -> tuple[list[np.ndarray], np.ndarray]:
    """Training input for one position: 4 index arrays (mover-relative order) + globals."""
    k = mover_index(state)
    per_seat = perspective_indices(state.board)
    return [per_seat[(k + i) % 4] for i in range(4)], global_features(state, k)


def collate_nnue(xs: list, device: str = "cpu"):
    bags = [b for idx, _ in xs for b in idx]
    lengths = [len(b) for b in bags]
    indices = torch.from_numpy(np.concatenate(bags).astype(np.int64))
    offsets = torch.tensor([0] + list(np.cumsum(lengths)[:-1]), dtype=torch.long)
    globals_ = torch.from_numpy(np.stack([g for _, g in xs]))
    return indices.to(device), offsets.to(device), globals_.to(device)


def pairwise_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


# --- weights ----------------------------------------------------------------------------

def export_weights(model: NNUE) -> dict[str, np.ndarray]:
    """Weight file contents (.npz): ft_w (1600, H), ft_b (H), l{i}_w (out, in) and l{i}_b
    for i = 1..n, residual (n bools), out_w (6, last), out_b (6). All float32."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    out = {"ft_w": sd["ft.weight"], "ft_b": sd["ft_bias"], "out_w": sd["out.weight"],
           "out_b": sd["out.bias"], "residual": np.array(model.skip, dtype=bool)}
    for i in range(len(model.layers)):
        out[f"l{i + 1}_w"] = sd[f"layers.{i}.weight"]
        out[f"l{i + 1}_b"] = sd[f"layers.{i}.bias"]
    return out


def save_weights(model: NNUE, path: str | Path) -> None:
    np.savez(path, **export_weights(model))


try:                                    # the compiled core (engine/), optional
    import chaturaji_engine as _engine
except ImportError:
    _engine = None

CORE_AVAILABLE = _engine is not None


# --- NumPy reference (search) -----------------------------------------------------------

class NumpyNNUE:
    """The same network as NNUE, as float32 NumPy arrays. With `use_core` and the compiled
    core installed, it also holds the core's copy of the weights (`self.core`)."""

    def __init__(self, weights: dict[str, np.ndarray], use_core: bool = True):
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
        self.ft_w, self.ft_b = f32(weights["ft_w"]), f32(weights["ft_b"])
        n = sum(1 for k in weights if k[0] == "l" and k[1:-2].isdigit() and k.endswith("_w"))
        self.layers = [(f32(weights[f"l{i}_w"]), f32(weights[f"l{i}_b"])) for i in range(1, n + 1)]
        res = weights.get("residual")
        self.residual = [bool(x) for x in res] if res is not None else [False] * n
        self.out_w, self.out_b = f32(weights["out_w"]), f32(weights["out_b"])
        self.core = None
        if use_core and _engine is not None:
            self.core = _engine.Nnue(self.ft_w, self.ft_b, [w for w, _ in self.layers],
                                     [b for _, b in self.layers], self.residual, self.out_w, self.out_b)

    @classmethod
    def from_model(cls, model: NNUE, use_core: bool = True) -> "NumpyNNUE":
        return cls(export_weights(model), use_core)

    @classmethod
    def load(cls, path: str | Path, use_core: bool = True) -> "NumpyNNUE":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files}, use_core)

    @property
    def num_params(self) -> int:
        return int(self.ft_w.size + self.ft_b.size + sum(w.size + b.size for w, b in self.layers)
                   + self.out_w.size + self.out_b.size)

    def refresh(self, board: Board) -> np.ndarray:
        """(4, H) accumulators (absolute seats) computed from scratch."""
        return np.stack([self.ft_b + self.ft_w[idx].sum(0) for idx in perspective_indices(board)])

    def forward(self, acc: np.ndarray, k: int, globals_: np.ndarray) -> np.ndarray:
        h = acc[_REL_ORDER[k]].reshape(-1)
        h = np.concatenate([np.minimum(np.maximum(h, 0.0), 1.0), globals_])
        for (w, b), skip in zip(self.layers, self.residual):
            y = np.minimum(np.maximum(w @ h + b, 0.0), 1.0)
            h = y + h if skip else y
        return self.out_w @ h + self.out_b

    def logits(self, acc: np.ndarray, state: ChaturajiState) -> np.ndarray:
        k = mover_index(state)
        return self.forward(acc, k, global_features(state, k))


class Accumulator:
    """Four perspective accumulators that follow a game move by move.

    `push(new_board)` applies the board difference since the current board (a normal move
    changes 2-3 squares; an elimination rewrites a whole army as dead pieces), and `pop()`
    goes back, so a depth-first search keeps one stack instead of copying positions.
    Runs in the compiled core when the net has one (and `use_core`), else in NumPy.
    """

    def __init__(self, net: NumpyNNUE, board: Board, use_core: bool = True):
        self.net = net
        self.core = net.core.accumulator(board_codes(board)) if (use_core and net.core) else None
        if self.core is None:
            self.stack = [(net.refresh(board), [row[:] for row in board])]

    @property
    def acc(self) -> np.ndarray:
        return np.asarray(self.core.acc()) if self.core is not None else self.stack[-1][0]

    def push(self, new_board: Board) -> None:
        if self.core is not None:
            self.core.push(board_codes(new_board))
            return
        acc, board = self.stack[-1]
        removed, added = board_diff(board, new_board)
        acc = acc.copy()
        for piece, r, c in removed:
            acc -= self.net.ft_w[piece_features(piece, r, c)]   # one row per perspective
        for piece, r, c in added:
            acc += self.net.ft_w[piece_features(piece, r, c)]
        self.stack.append((acc, [row[:] for row in new_board]))

    def pop(self) -> None:
        if self.core is not None:
            self.core.pop()
            return
        if len(self.stack) == 1:
            raise IndexError("pop from the root position")
        self.stack.pop()

    def logits(self, state: ChaturajiState) -> np.ndarray:
        k = mover_index(state)
        if self.core is not None:
            return np.asarray(self.core.logits(k, global_features(state, k)))
        return self.net.forward(self.stack[-1][0], k, global_features(state, k))

    def evaluate(self, state: ChaturajiState) -> np.ndarray:
        """Each seat's expected FFA score, absolute order (R, B, Y, G)."""
        p = 1.0 / (1.0 + np.exp(-self.logits(state)))
        return utilities(p)[_TO_ABS[mover_index(state)]]

    def evaluate_children(self, children: list[ChaturajiState]) -> list[np.ndarray]:
        """`evaluate` for several positions one move (or a few board changes) from the
        current one, in one batched pass through the compiled core: each weight is read
        once for all of them, which is what makes the large presets affordable."""
        if not children:
            return []
        ks = [mover_index(c) for c in children]
        if self.core is not None:
            boards = np.stack([board_codes(c.board) for c in children])
            g = np.stack([global_features(c, k) for c, k in zip(children, ks)])
            logits = np.asarray(self.core.child_logits(boards, ks, g))
        else:
            logits = []
            for c in children:
                self.push(c.board)
                logits.append(self.logits(c))
                self.pop()
            logits = np.stack(logits)
        u = utilities(1.0 / (1.0 + np.exp(-logits)))
        return [u[i][_TO_ABS[k]] for i, k in enumerate(ks)]


# --- MCTS adapter (lets the NNUE play through the existing evaluation tools) ------------

class NNUEEvaluator:
    """bot.mcts evaluator: uniform priors (no policy head), NNUE utilities as the value."""

    def __init__(self, model: NNUE, device: str = "cpu"):
        self.model, self.device = model, device

    @torch.no_grad()
    def __call__(self, state: ChaturajiState):
        from .features import legal_canonical
        moves, actions = legal_canonical(state)
        logits = self.model(*collate_nnue([nnue_features(state)], self.device))
        u = utilities(torch.sigmoid(logits[0]).cpu().numpy())
        priors = np.full(len(moves), 1.0 / max(len(moves), 1), dtype=np.float32)
        # FFA utilities sum to 2 over the seats; halve them to the 0-1 total MCTS backs up.
        return moves, actions, priors, to_absolute(u / 2.0, mover_index(state))
