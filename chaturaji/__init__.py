"""Chaturaji (4-player chess variant) game state and rules."""

from .constants import COLORS, DEAD, PIECE_POINTS, TURN_ORDER
from .game import MoveInfo, apply_action, apply_move, eliminate, game_over_reason, is_game_over, winners
from .moves import RESIGN, Move, Resign, legal_moves
from .record import Game
from .state import ChaturajiState, index_to_square, square_to_index

__all__ = [
    "COLORS", "DEAD", "PIECE_POINTS", "TURN_ORDER",
    "MoveInfo", "apply_action", "apply_move", "eliminate", "game_over_reason", "is_game_over", "winners",
    "RESIGN", "Move", "Resign", "legal_moves", "Game", "ChaturajiState", "index_to_square", "square_to_index",
]
