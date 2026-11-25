"""Basic structure for the MCTS evaluator: encoding layer -> 3 fully connected layers -> evaluation.

This is a skeleton to build on (see bot/PLAN.md section 4). It works end to end and
plugs into the existing MCTS through `MLPEvaluator`.

    state --sparse_features--> (piece-square indices, global features)
          --ChaturajiMLP-->     policy logits (4096), value (4 seats), score (4 seats)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from chaturaji.constants import DEAD, PIECE_TYPES, TURN_ORDER
from chaturaji.state import ChaturajiState, board_key

from .features import (
    NUM_ACTIONS, legal_canonical, mover_index, relative_owner, rotate_square, to_absolute,
)

NUM_PIECE_FEATURES = 5 * len(PIECE_TYPES) * 64   # (relative owner, type, square) = 1600
NUM_GLOBALS = 16
_TYPE_IDX = {t: i for i, t in enumerate(PIECE_TYPES)}


def sparse_features(state: ChaturajiState) -> tuple[np.ndarray, np.ndarray]:
    """Active piece-square feature indices + dense global features, from the mover's view."""
    k = mover_index(state)
    idx = []
    for r, row in enumerate(state.board):
        for c, p in enumerate(row):
            if p is not None:
                cr, cc = rotate_square(r, c, k)
                idx.append((relative_owner(p[0], k) * 5 + _TYPE_IDX[p[1]]) * 64 + cr * 8 + cc)
    g = global_features(state, k)
    return np.array(idx, dtype=np.int64), g


def global_features(state: ChaturajiState, k: int | None = None) -> np.ndarray:
    """The 16 dense global features, seats ordered from seat k's view (default: the mover)."""
    if k is None:
        k = mover_index(state)
    seats = [TURN_ORDER[(k + i) % 4] for i in range(4)]
    scores = np.array([state.scores[s] for s in seats], dtype=np.float32)
    return np.concatenate([
        [0.0 if s in state.eliminated else 1.0 for s in seats],      # alive
        scores / 30.0,                                                # scores
        (scores - scores.mean()) / 20.0,                              # score vs mean
        [state.position_counts.get(board_key(state.board), 0) / 3.0,  # repetition
         min(state.turn / 400.0, 1.0),                                # progress
         sum(row.count(DEAD + "K") for row in state.board) / 3.0,     # dead kings
         len(state.active_players()) / 4.0],                          # players alive
    ]).astype(np.float32)


class ChaturajiMLP(nn.Module):
    def __init__(self, embed: int = 512, widths: tuple[int, int, int] = (1024, 512, 256),
                 dropout: float = 0.1):
        super().__init__()
        # Encoding layer: one learned vector per (owner, piece, square), summed over the board.
        self.encoder = nn.EmbeddingBag(NUM_PIECE_FEATURES, embed, mode="sum")
        self.encoder_bias = nn.Parameter(torch.zeros(embed))
        w1, w2, w3 = widths
        # Three fully connected layers.
        self.fc = nn.Sequential(
            nn.Linear(embed + NUM_GLOBALS, w1), nn.LayerNorm(w1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(w1, w2), nn.LayerNorm(w2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(w2, w3), nn.LayerNorm(w3), nn.ReLU(),
        )
        # Evaluation outputs.
        self.value = nn.Linear(w3, 4)            # P(top score) per relative seat (softmax)
        self.score = nn.Linear(w3, 4)            # final score margin per relative seat
        self.policy = nn.Linear(w3, NUM_ACTIONS)  # move priors for MCTS

    def forward(self, indices, offsets, globals_):
        h = torch.clamp(self.encoder(indices, offsets) + self.encoder_bias, 0.0, 1.0)
        h = self.fc(torch.cat([h, globals_], dim=1))
        return self.policy(h), self.value(h), self.score(h)


def batch_features(states: list[ChaturajiState], device: str = "cpu"):
    """Pack several states for one forward pass (EmbeddingBag indices + offsets)."""
    return collate_sparse([sparse_features(s) for s in states], device)


def collate_sparse(feats: list[tuple[np.ndarray, np.ndarray]], device: str = "cpu"):
    """Pack precomputed (indices, globals) pairs into (indices, offsets, globals) tensors."""
    lengths = [len(i) for i, _ in feats]
    indices = torch.from_numpy(np.concatenate([i for i, _ in feats]))
    offsets = torch.tensor([0] + list(np.cumsum(lengths)[:-1]), dtype=torch.long)
    globals_ = torch.from_numpy(np.stack([g for _, g in feats]))
    return indices.to(device), offsets.to(device), globals_.to(device)


class MLPEvaluator:
    """Drop-in evaluator for bot.mcts.MCTS: returns (moves, actions, priors, value_abs)."""

    def __init__(self, model: ChaturajiMLP, device: str = "cpu"):
        self.model, self.device = model, device

    @torch.no_grad()
    def __call__(self, state: ChaturajiState):
        moves, actions = legal_canonical(state)
        logits, value_logits, _ = self.model(*batch_features([state], self.device))
        priors = torch.softmax(logits[0, torch.from_numpy(actions).to(self.device)], 0).cpu().numpy()
        v_rel = torch.softmax(value_logits[0], 0).cpu().numpy()
        return moves, actions, priors, to_absolute(v_rel, mover_index(state))

    # TODO (guidance welcome): batched evaluation with virtual loss, an LRU cache,
    # and a heuristic-prior fallback for a value-only model.
