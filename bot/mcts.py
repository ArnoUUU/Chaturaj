"""Multi-player (max^n) PUCT Monte Carlo Tree Search.

Values are 4-vectors in absolute seat order (R, B, Y, G), each in [0, 1]: the estimated
share of "finishes with the top score". At each node the player to move picks the child
that maximises its own entry.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from chaturaji.constants import TURN_ORDER
from chaturaji.game import apply_action, eliminate, is_game_over
from chaturaji.moves import RESIGN, Move, Resign
from chaturaji.state import ChaturajiState

from .features import encode, legal_canonical, mover_index, outcome_vectors, to_absolute


class UniformEvaluator:
    """No network: uniform priors and a neutral value. Useful for tests and baselines."""

    def __call__(self, state: ChaturajiState):
        moves, idx = legal_canonical(state)
        priors = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
        return moves, idx, priors, np.full(4, 0.25, dtype=np.float32)


class NetEvaluator:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def __call__(self, state: ChaturajiState):
        moves, idx = legal_canonical(state)
        x = torch.from_numpy(encode(state)).unsqueeze(0).to(self.device)
        logits, value_logits, _ = self.model(x)
        legal_logits = logits[0, torch.from_numpy(idx).to(self.device)]
        priors = torch.softmax(legal_logits, 0).cpu().numpy()
        v_rel = torch.softmax(value_logits[0], 0).cpu().numpy()
        return moves, idx, priors, to_absolute(v_rel, mover_index(state))


class Node:
    __slots__ = ("state", "mover", "moves", "actions", "priors", "children",
                 "child_n", "child_w", "value", "terminal")

    def __init__(self, state: ChaturajiState):
        self.state = state
        self.mover = TURN_ORDER.index(state.current_player())
        self.moves: list[Move] | None = None
        self.actions: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.children: list[Node | None] = []
        self.child_n: np.ndarray | None = None
        self.child_w: np.ndarray | None = None
        self.value: np.ndarray | None = None
        self.terminal = False

    @property
    def expanded(self) -> bool:
        return self.moves is not None


SELECTIONS = ("puct", "ucb1")


class MCTS:
    """Tree search with a choice of selection rule.

    selection="puct": AlphaZero rule, Q + c_puct * P * sqrt(N) / (1 + n). Uses the
                      network's move priors P.
    selection="ucb1": classic Upper Confidence Bound, Q + c_ucb * sqrt(ln N / n). Every
                      child is tried once first (highest prior first), so priors only
                      break ties. Works with a value-only evaluator.
    Q is always the value for the player to move at that node (max^n).
    """

    def __init__(self, evaluator, num_simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, noise_eps: float = 0.25,
                 max_turn: int | None = 400, rng: np.random.Generator | None = None,
                 selection: str = "puct", c_ucb: float = math.sqrt(2),
                 resign: bool = True, resign_prior: float = 0.02):
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.selection = selection
        self.c_ucb = c_ucb
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_eps = noise_eps
        self.max_turn = max_turn
        self.rng = rng or np.random.default_rng()
        self.resign = resign
        self.resign_prior = resign_prior

    def is_terminal(self, state: ChaturajiState) -> bool:
        return is_game_over(state) or (self.max_turn is not None and state.turn >= self.max_turn)

    def _evaluate(self, node: Node, root: bool = False) -> np.ndarray:
        if self.is_terminal(node.state):
            node.terminal = True
            node.value = outcome_vectors(node.state)[0]
            return node.value
        moves, actions, priors, value = self.evaluator(node.state)
        node.moves, node.actions, node.priors = moves, actions, priors.astype(np.float64)
        node.children = [None] * len(moves)
        node.child_n = np.zeros(len(moves), dtype=np.float64)
        node.child_w = np.zeros((len(moves), 4), dtype=np.float64)
        node.value = value
        if self._wants_resign(node, root):
            self._add_resign(node)
        return value

    def _wants_resign(self, node: Node, root: bool) -> bool:
        """Resign is a child at every two-player node, and at the search root with three
        or more players (deeper three-player nodes skip it, to keep the tree small)."""
        if not self.resign:
            return False
        active = len(node.state.active_players())
        return active == 2 or (active >= 3 and root)

    @staticmethod
    def _has_resign(node: Node) -> bool:
        return bool(node.moves) and isinstance(node.moves[-1], Resign)

    def _add_resign(self, node: Node) -> None:
        """Append the resign child to an expanded node (once). Its prior is `resign_prior`,
        the move priors shrink to make room, and it starts with one visit at its value so
        it is always compared, even with few simulations."""
        if self._has_resign(node):
            return
        child = Node(eliminate(node.state, node.state.current_player()))
        value = self._evaluate(child)  # terminal (exact outcome) or one network call
        p = self.resign_prior
        node.moves = list(node.moves) + [RESIGN]
        node.actions = np.append(node.actions, -1)
        node.priors = np.concatenate([node.priors * (1 - p), [p]])
        node.children = node.children + [child]
        node.child_n = np.append(node.child_n, 1.0)
        node.child_w = np.vstack([node.child_w, np.asarray(value, dtype=np.float64)[None, :]])

    def _select(self, node: Node) -> int:
        if self.selection == "ucb1":
            return self._select_ucb1(node)
        return self._select_puct(node)

    def _select_ucb1(self, node: Node) -> int:
        unvisited = np.flatnonzero(node.child_n == 0)
        if len(unvisited):
            return int(unvisited[np.argmax(node.priors[unvisited])])
        n_total = node.child_n.sum()
        q = node.child_w[:, node.mover] / node.child_n
        u = self.c_ucb * np.sqrt(math.log(n_total) / node.child_n)
        return int(np.argmax(q + u))

    def _select_puct(self, node: Node) -> int:
        n_total = node.child_n.sum()
        visited = node.child_n > 0
        q = np.where(visited, node.child_w[:, node.mover] / np.maximum(node.child_n, 1),
                     node.value[node.mover])  # first-play urgency = parent estimate
        u = self.c_puct * node.priors * math.sqrt(n_total + 1) / (1 + node.child_n)
        return int(np.argmax(q + u))

    def search(self, root: Node, add_noise: bool = False, time_limit: float | None = None,
               stop=None, early_stop: bool = True) -> Node:
        """Run simulations from `root`.

        Without `time_limit`, runs exactly `num_simulations`. With `time_limit` (seconds),
        searches until the time is used up, or until the most-visited move can no longer be
        overtaken at the current speed (early stop, bot/PLAN.md section 5.7);
        `num_simulations` is then a cap (None = no cap). `stop` is an optional callable
        (e.g. an Event's `is_set`) checked between simulations to abort early. The root
        always gets at least one simulation so `choose` has a move.
        """
        if not root.expanded and not root.terminal:
            self._evaluate(root, root=True)
        if root.terminal:
            return root
        if self._wants_resign(root, root=True):
            self._add_resign(root)  # a root expanded earlier as an inner node (tree reuse)
        if add_noise:
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(root.priors))
            root.priors = (1 - self.noise_eps) * root.priors + self.noise_eps * noise
        deadline = None if time_limit is None else time.monotonic() + time_limit
        cap = self.num_simulations
        if cap is None and deadline is None and stop is None:
            raise ValueError("num_simulations=None needs a time_limit or a stop callback")
        done, t0 = 0, time.monotonic()
        while cap is None or done < cap:
            if done and stop is not None and stop():
                break
            if done and deadline is not None:
                now = time.monotonic()
                if now >= deadline:
                    break
                if early_stop and done % 32 == 0 and len(root.child_n) > 1:
                    left = (deadline - now) * done / max(now - t0, 1e-9)
                    if cap is not None:
                        left = min(left, cap - done)
                    top2 = np.partition(root.child_n, -2)[-2:]
                    if top2[1] - top2[0] > left:
                        break
            done += 1
            node, path = root, []
            while node.expanded and not node.terminal:
                i = self._select(node)
                path.append((node, i))
                child = node.children[i]
                if child is None:
                    child = Node(apply_action(node.state, node.moves[i])[0])
                    node.children[i] = child
                node = child
            value = node.value if node.terminal else self._evaluate(node)
            for n, i in path:
                n.child_n[i] += 1
                n.child_w[i] += value
        self.last_simulations = done
        return root

    @staticmethod
    def advance(root: Node | None, move: Move) -> Node | None:
        """Tree reuse: the already-searched child of `root` reached by `move`, or None."""
        if root is None or not root.expanded:
            return None
        for i, m in enumerate(root.moves):
            if m == move:
                return root.children[i]
        return None

    @staticmethod
    def _num_moves(root: Node) -> int:
        """Number of move children (the resign child, if any, is always last)."""
        return len(root.moves) - (1 if MCTS._has_resign(root) else 0)

    @staticmethod
    def real_visits(root: Node) -> int:
        """Simulations through `root`, not counting the resign child's one pre-visit."""
        return int(root.child_n.sum()) - (1 if MCTS._has_resign(root) else 0)

    @staticmethod
    def move_actions(root: Node) -> np.ndarray:
        """Canonical action indices of the move children (the policy head has no resign)."""
        return root.actions[:MCTS._num_moves(root)]

    @staticmethod
    def visit_policy(root: Node) -> np.ndarray:
        """Visit distribution over the move children (no resign entry), renormalised. If the
        simulations all went to the resign child (no move was visited), the renormalised
        move priors stand in for it."""
        n = MCTS._num_moves(root)
        counts = root.child_n[:n]
        if counts.sum() > 0:
            return counts / counts.sum()
        priors = root.priors[:n]
        return priors / priors.sum()

    def choose(self, root: Node, temperature: float = 0.0) -> int:
        """Index of the chosen child. The resign child's count is its visits minus the one
        pre-visit. Ties go to a move (resign is last and `argmax` takes the first maximum).
        A resignation that wins outright (two players left, a big enough lead) is always
        chosen, like max^n's immediate-win cut.
        With temperature > 0, resign is returned only if it is the temperature-0 choice;
        otherwise the sample is over the move children only."""
        counts = root.child_n.copy()
        if MCTS._has_resign(root):
            r = len(root.moves) - 1
            claim = root.children[r]
            if claim is not None and claim.terminal and claim.value[root.mover] >= 1.0 - 1e-9:
                return r        # resigning wins outright (a claimed win): nothing beats that
            counts[-1] -= 1.0
        best = int(np.argmax(counts))
        if temperature <= 1e-6:
            return best
        n_moves = MCTS._num_moves(root)
        if best >= n_moves:  # resign is the best child
            return best
        p = counts[:n_moves] ** (1.0 / temperature)
        return int(self.rng.choice(n_moves, p=p / p.sum()))
