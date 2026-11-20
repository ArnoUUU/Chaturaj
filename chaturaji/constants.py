"""Static rule data for Chaturaji (chess.com rules, see PLAN.md)."""

BOARD_SIZE = 8

# Turn order: Red moves first.
TURN_ORDER = ("R", "B", "Y", "G")
COLORS = TURN_ORDER
PIECE_TYPES = ("K", "B", "N", "R", "P")

# Owner code for dead (colorless) pieces: once a color's king is captured, all of that
# color's remaining pieces are rewritten as "X" + type. They block, can be captured by
# anyone, never move, and are worth 0 points.
DEAD = "X"
OWNERS = COLORS + (DEAD,)

# Points for capturing an *active* piece. Dead pieces are worth 0.
PIECE_POINTS = {"P": 1, "N": 3, "K": 3, "B": 5, "R": 5}

# Bonus for a move that gives check and leaves the mover attacking this many active
# enemy kings at once. Check never forces a king to move; it only scores.
CHECK_BONUS = {2: 1, 3: 5}

# Board indexing: board[row][col], row 0 = rank 8 (top), row 7 = rank 1 (bottom, Red's side),
# col 0 = file a.
# Pawn forward direction as (drow, dcol). A pawn promotes to a rook (boat) when it reaches
# the far edge in its direction, i.e. the 8th rank relative to its own side.
PAWN_DIRECTION = {
    "R": (-1, 0),  # up
    "B": (0, 1),   # right
    "Y": (1, 0),   # down
    "G": (0, -1),  # left
}

KNIGHT_OFFSETS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
ROOK_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BISHOP_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Starting position in algebraic squares. Red (bottom row) is confirmed; the other
# three corners are assumed and can be changed here.
#
#       a    b    c    d    e    f    g    h
#  8    BR   BP   .    .    YK   YB   YN   YR
#  7    BN   BP   .    .    YP   YP   YP   YP
#  6    BB   BP   .    .    .    .    .    .
#  5    BK   BP   .    .    .    .    .    .
#  4    .    .    .    .    .    .    GP   GK
#  3    .    .    .    .    .    .    GP   GB
#  2    RP   RP   RP   RP   .    .    GP   GN
#  1    RR   RN   RB   RK   .    .    GP   GR
STARTING_LAYOUT = {
    "R": {"a1": "R", "b1": "N", "c1": "B", "d1": "K",
          "a2": "P", "b2": "P", "c2": "P", "d2": "P"},
    "B": {"a8": "R", "a7": "N", "a6": "B", "a5": "K",
          "b8": "P", "b7": "P", "b6": "P", "b5": "P"},
    "Y": {"h8": "R", "g8": "N", "f8": "B", "e8": "K",
          "h7": "P", "g7": "P", "f7": "P", "e7": "P"},
    "G": {"h1": "R", "h2": "N", "h3": "B", "h4": "K",
          "g1": "P", "g2": "P", "g3": "P", "g4": "P"},
}
