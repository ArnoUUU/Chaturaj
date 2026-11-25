//! Chaturaji rules (chess.com four-player variant), ported from the Python package
//! `chaturaji/` (moves.py, game.py), which stays the reference. `bot/rules_core.py` replays
//! every scraped game in both and compares them at every ply.
//!
//! Board: 64 bytes, square = row * 8 + col (row 0 = rank 8, col 0 = file a), 0 = empty,
//! else 1 + owner * 5 + type with owners R B Y G X (dead) = 0..4 and types K B N R P = 0..4
//! (the same codes as the NNUE core). Turn order R, B, Y, G; eliminated seats are skipped.
//!
//! Rules (see chaturaji/constants.py):
//! - no check restriction: every pseudo-legal move is legal, kings can be captured
//! - capturing an active piece scores P 1, N 3, K 3, B 5, R 5; capturing a king eliminates
//!   that player, whose pieces all become dead (they block, never move, score 0 when
//!   captured, except a dead king: 3)
//! - a move that gives a new check and leaves the mover attacking 2 or 3 active kings
//!   scores 1 or 5
//! - pawns move one square forward (each color has its own direction), capture diagonally
//!   forward, and promote to a rook on the far edge
//! - the game ends when one active player is left (who gets 3 per dead king on the board)
//!   or a board position occurs a third time (every active player gets 2)
//!
//! `Position::push` / `pop` make and unmake moves for search; `eliminate` handles
//! resignations and timeouts (also undoable).

use std::collections::HashMap;

pub const EMPTY: u8 = 0;
pub const DEAD: usize = 4;
pub const KING: usize = 0;
pub const BISHOP: usize = 1;
pub const KNIGHT: usize = 2;
pub const ROOK: usize = 3;
pub const PAWN: usize = 4;
const PIECE_POINTS: [i32; 5] = [3, 5, 3, 5, 1]; // K B N R P
const DEAD_KING_POINTS: i32 = 3;
const LAST_STANDING_PER_DEAD_KING: i32 = 3;
const REPETITION_POINTS: i32 = 2;

const KNIGHT_OFFSETS: [(i32, i32); 8] = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
const KING_OFFSETS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
const ROOK_DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const PAWN_DIRECTION: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)]; // R up, B right, Y down, G left

#[inline]
pub fn code(owner: usize, ty: usize) -> u8 {
    (1 + owner * 5 + ty) as u8
}

#[inline]
pub fn owner_of(c: u8) -> usize {
    (c as usize - 1) / 5
}

#[inline]
pub fn type_of(c: u8) -> usize {
    (c as usize - 1) % 5
}

#[inline]
fn on_board(r: i32, c: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&c)
}

/// A move: from-square * 64 + to-square. Promotion is forced, so it isn't encoded.
pub type Move = u16;

#[inline]
pub fn mv(from: usize, to: usize) -> Move {
    (from * 64 + to) as Move
}

#[inline]
pub fn from_sq(m: Move) -> usize {
    m as usize / 64
}

#[inline]
pub fn to_sq(m: Move) -> usize {
    m as usize % 64
}

pub fn is_promotion_square(color: usize, sq: usize) -> bool {
    let (dr, dc) = PAWN_DIRECTION[color];
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    if dr != 0 {
        r == if dr < 0 { 0 } else { 7 }
    } else {
        c == if dc < 0 { 0 } else { 7 }
    }
}

fn pawn_capture_offsets(color: usize) -> [(i32, i32); 2] {
    let (dr, dc) = PAWN_DIRECTION[color];
    if dr != 0 {
        [(dr, -1), (dr, 1)]
    } else {
        [(-1, dc), (1, dc)]
    }
}

/// Moves of the piece on `sq`, in the same order as chaturaji.moves.piece_moves.
fn piece_moves(board: &[u8; 64], sq: usize, out: &mut Vec<Move>) {
    let p = board[sq];
    if p == EMPTY || owner_of(p) == DEAD {
        return;
    }
    let (color, ty) = (owner_of(p), type_of(p));
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    let target_ok = |t: u8| t == EMPTY || owner_of(t) != color;
    match ty {
        ROOK | BISHOP => {
            let dirs = if ty == ROOK { &ROOK_DIRS } else { &BISHOP_DIRS };
            for &(dr, dc) in dirs {
                let (mut tr, mut tc) = (r + dr, c + dc);
                while on_board(tr, tc) {
                    let t = board[(tr * 8 + tc) as usize];
                    if t == EMPTY {
                        out.push(mv(sq, (tr * 8 + tc) as usize));
                    } else {
                        if owner_of(t) != color {
                            out.push(mv(sq, (tr * 8 + tc) as usize));
                        }
                        break;
                    }
                    tr += dr;
                    tc += dc;
                }
            }
        }
        KNIGHT | KING => {
            let offs = if ty == KNIGHT { &KNIGHT_OFFSETS } else { &KING_OFFSETS };
            for &(dr, dc) in offs {
                let (tr, tc) = (r + dr, c + dc);
                if on_board(tr, tc) && target_ok(board[(tr * 8 + tc) as usize]) {
                    out.push(mv(sq, (tr * 8 + tc) as usize));
                }
            }
        }
        _ => {
            let (dr, dc) = PAWN_DIRECTION[color];
            let (tr, tc) = (r + dr, c + dc);
            if on_board(tr, tc) && board[(tr * 8 + tc) as usize] == EMPTY {
                out.push(mv(sq, (tr * 8 + tc) as usize));
            }
            for (odr, odc) in pawn_capture_offsets(color) {
                let (tr, tc) = (r + odr, c + odc);
                if on_board(tr, tc) {
                    let t = board[(tr * 8 + tc) as usize];
                    if t != EMPTY && owner_of(t) != color {
                        out.push(mv(sq, (tr * 8 + tc) as usize));
                    }
                }
            }
        }
    }
}

/// True if any piece of `color` attacks `sq` (chaturaji.moves.is_attacked_by).
pub fn is_attacked_by(board: &[u8; 64], sq: usize, color: usize) -> bool {
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    let at = |tr: i32, tc: i32| board[(tr * 8 + tc) as usize];
    for &(dr, dc) in &KNIGHT_OFFSETS {
        if on_board(r + dr, c + dc) && at(r + dr, c + dc) == code(color, KNIGHT) {
            return true;
        }
    }
    for &(dr, dc) in &KING_OFFSETS {
        if on_board(r + dr, c + dc) && at(r + dr, c + dc) == code(color, KING) {
            return true;
        }
    }
    for (dirs, ty) in [(&ROOK_DIRS, ROOK), (&BISHOP_DIRS, BISHOP)] {
        for &(dr, dc) in dirs {
            let (mut tr, mut tc) = (r + dr, c + dc);
            while on_board(tr, tc) {
                let t = at(tr, tc);
                if t != EMPTY {
                    if t == code(color, ty) {
                        return true;
                    }
                    break;
                }
                tr += dr;
                tc += dc;
            }
        }
    }
    for (odr, odc) in pawn_capture_offsets(color) {
        let (tr, tc) = (r - odr, c - odc);
        if on_board(tr, tc) && at(tr, tc) == code(color, PAWN) {
            return true;
        }
    }
    false
}

fn find_king(board: &[u8; 64], color: usize) -> Option<usize> {
    board.iter().position(|&p| p == code(color, KING))
}

/// Bitmask of active enemy colors whose king `color` attacks.
fn kings_attacked_by(board: &[u8; 64], color: usize, eliminated: u8) -> u8 {
    let mut out = 0u8;
    for enemy in 0..4 {
        if enemy == color || eliminated & (1 << enemy) != 0 {
            continue;
        }
        if let Some(k) = find_king(board, enemy) {
            if is_attacked_by(board, k, color) {
                out |= 1 << enemy;
            }
        }
    }
    out
}

// --- Zobrist hashing (board only, for repetition) ---------------------------------------

pub const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

const fn zobrist_table() -> [[u64; 26]; 64] {
    let mut t = [[0u64; 26]; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut c = 1;
        while c < 26 {
            t[sq][c] = splitmix((sq * 26 + c) as u64 * 0x1000193 + 7);
            c += 1;
        }
        sq += 1;
    }
    t
}

static ZOBRIST: [[u64; 26]; 64] = zobrist_table();

pub fn board_hash(board: &[u8; 64]) -> u64 {
    board.iter().enumerate().filter(|(_, &p)| p != EMPTY).fold(0, |h, (sq, &p)| h ^ ZOBRIST[sq][p as usize])
}

// --- positions -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOver {
    LastKing,
    Repetition,
}

/// What a move did (chaturaji.game.MoveInfo, minus the notation fields).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveInfo {
    pub points: i32,       // all points the mover gained, end bonus included
    pub captured: u8,      // captured piece code (0 = none)
    pub check_bonus: i32,
    pub eliminated: Option<usize>,
    /// After the move the mover attacks an active king it did not attack before.
    pub gives_check: bool,
}

#[derive(Clone)]
struct Undo {
    board: [u8; 64],
    turn: u32,
    scores: [i32; 4],
    eliminated: u8,
    hash: u64,
    counted: bool, // the move added one to counts[hash after the move]
    over: Option<GameOver>,
}

#[derive(Clone)]
pub struct Position {
    pub board: [u8; 64],
    pub turn: u32,
    pub scores: [i32; 4],
    pub eliminated: u8, // bitmask over R B Y G
    pub hash: u64,      // board hash
    counts: HashMap<u64, u32>,
    over: Option<GameOver>,
    undo: Vec<Undo>,
}

pub fn starting_board() -> [u8; 64] {
    // squares as (row, col): row 0 = rank 8
    let mut b = [EMPTY; 64];
    let put = |b: &mut [u8; 64], sq: &str, owner: usize, ty: usize| {
        let bytes = sq.as_bytes();
        let col = (bytes[0] - b'a') as usize;
        let row = 8 - (bytes[1] - b'0') as usize;
        b[row * 8 + col] = code(owner, ty);
    };
    let layout: [(usize, [(&str, usize); 8]); 4] = [
        (0, [("a1", ROOK), ("b1", KNIGHT), ("c1", BISHOP), ("d1", KING), ("a2", PAWN), ("b2", PAWN), ("c2", PAWN), ("d2", PAWN)]),
        (1, [("a8", ROOK), ("a7", KNIGHT), ("a6", BISHOP), ("a5", KING), ("b8", PAWN), ("b7", PAWN), ("b6", PAWN), ("b5", PAWN)]),
        (2, [("h8", ROOK), ("g8", KNIGHT), ("f8", BISHOP), ("e8", KING), ("h7", PAWN), ("g7", PAWN), ("f7", PAWN), ("e7", PAWN)]),
        (3, [("h1", ROOK), ("h2", KNIGHT), ("h3", BISHOP), ("h4", KING), ("g1", PAWN), ("g2", PAWN), ("g3", PAWN), ("g4", PAWN)]),
    ];
    for (owner, pieces) in layout {
        for (sq, ty) in pieces {
            put(&mut b, sq, owner, ty);
        }
    }
    b
}

impl Position {
    pub fn initial() -> Self {
        let board = starting_board();
        let hash = board_hash(&board);
        let mut counts = HashMap::new();
        counts.insert(hash, 1);
        Position { board, turn: 0, scores: [0; 4], eliminated: 0, hash, counts, over: None, undo: Vec::new() }
    }

    /// A position from its parts. `counts` lists earlier boards and how often each occurred
    /// (chaturaji's position_counts), the current board included.
    pub fn from_parts(board: [u8; 64], turn: u32, scores: [i32; 4], eliminated: u8, counts: Vec<([u8; 64], u32)>) -> Result<Self, String> {
        if let Some(&c) = board.iter().find(|&&c| c > 25) {
            return Err(format!("invalid piece code {}", c));
        }
        let hash = board_hash(&board);
        let mut map = HashMap::new();
        for (b, n) in counts {
            *map.entry(board_hash(&b)).or_insert(0) += n;
        }
        let mut p = Position { board, turn, scores, eliminated, hash, counts: map, over: None, undo: Vec::new() };
        p.over = p.compute_over();
        Ok(p)
    }

    pub fn current_player(&self) -> usize {
        (self.turn % 4) as usize
    }

    pub fn active_count(&self) -> u32 {
        4 - self.eliminated.count_ones()
    }

    pub fn is_active(&self, color: usize) -> bool {
        self.eliminated & (1 << color) == 0
    }

    pub fn repetitions(&self) -> u32 {
        self.counts.get(&self.hash).copied().unwrap_or(0)
    }

    pub fn depth(&self) -> usize {
        self.undo.len()
    }

    fn compute_over(&self) -> Option<GameOver> {
        if self.active_count() <= 1 {
            Some(GameOver::LastKing)
        } else if self.counts.values().any(|&n| n >= 3) {
            Some(GameOver::Repetition)
        } else {
            None
        }
    }

    pub fn game_over(&self) -> Option<GameOver> {
        self.over
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::with_capacity(48);
        self.legal_moves_into(&mut out);
        out
    }

    pub fn legal_moves_into(&self, out: &mut Vec<Move>) {
        out.clear();
        let color = self.current_player();
        for sq in 0..64 {
            let p = self.board[sq];
            if p != EMPTY && owner_of(p) == color {
                piece_moves(&self.board, sq, out);
            }
        }
    }

    /// Points `m` would capture (-1 if it isn't a capture), for move ordering.
    pub fn capture_points(&self, m: Move) -> i32 {
        let t = self.board[to_sq(m)];
        if t == EMPTY {
            -1
        } else if owner_of(t) == DEAD {
            if type_of(t) == KING { DEAD_KING_POINTS } else { 0 }
        } else {
            PIECE_POINTS[type_of(t)]
        }
    }

    fn save(&self) -> Undo {
        Undo { board: self.board, turn: self.turn, scores: self.scores, eliminated: self.eliminated,
               hash: self.hash, counted: false, over: self.over }
    }

    fn kill(&mut self, color: usize) {
        self.eliminated |= 1 << color;
        for sq in 0..64 {
            let p = self.board[sq];
            if p != EMPTY && owner_of(p) == color {
                let d = code(DEAD, type_of(p));
                self.hash ^= ZOBRIST[sq][p as usize] ^ ZOBRIST[sq][d as usize];
                self.board[sq] = d;
            }
        }
    }

    fn advance_turn(&mut self) {
        self.turn += 1;
        if self.active_count() <= 1 {
            return;
        }
        while !self.is_active(self.current_player()) {
            self.turn += 1;
        }
    }

    /// End-of-game points if this transition ended the game; returns the mover's share.
    fn award_end_bonus(&mut self, mover: usize) -> i32 {
        self.over = self.compute_over_after();
        let per = match self.over {
            Some(GameOver::LastKing) => {
                let dead_kings = self.board.iter().filter(|&&p| p == code(DEAD, KING)).count() as i32;
                LAST_STANDING_PER_DEAD_KING * dead_kings
            }
            Some(GameOver::Repetition) => REPETITION_POINTS,
            None => 0,
        };
        let mut mine = 0;
        if per != 0 {
            for c in 0..4 {
                if self.is_active(c) {
                    self.scores[c] += per;
                    if c == mover {
                        mine = per;
                    }
                }
            }
        }
        mine
    }

    /// Game over after a transition: the repetition test only needs the current board,
    /// unless an earlier board had already reached 3 (then the game was over already).
    fn compute_over_after(&self) -> Option<GameOver> {
        if self.active_count() <= 1 {
            Some(GameOver::LastKing)
        } else if self.repetitions() >= 3 || self.undo.last().is_some_and(|u| u.over == Some(GameOver::Repetition)) {
            Some(GameOver::Repetition)
        } else {
            None
        }
    }

    /// Make a move (chaturaji.game.apply_move). Undo with `pop`.
    pub fn push(&mut self, m: Move) -> Result<MoveInfo, String> {
        let color = self.current_player();
        let (from, to) = (from_sq(m), to_sq(m));
        if from >= 64 || to >= 64 {
            return Err(format!("bad move {}", m));
        }
        let piece = self.board[from];
        if piece == EMPTY || owner_of(piece) != color {
            return Err(format!("move {}->{} does not move a piece of seat {}", from, to, color));
        }
        let before = kings_attacked_by(&self.board, color, self.eliminated);
        self.undo.push(self.save());

        let captured = self.board[to];
        let placed = if type_of(piece) == PAWN && is_promotion_square(color, to) { code(color, ROOK) } else { piece };
        if captured != EMPTY {
            self.hash ^= ZOBRIST[to][captured as usize];
        }
        self.hash ^= ZOBRIST[from][piece as usize] ^ ZOBRIST[to][placed as usize];
        self.board[to] = placed;
        self.board[from] = EMPTY;

        let mut info = MoveInfo { captured, ..Default::default() };
        if captured == code(DEAD, KING) {
            info.points += DEAD_KING_POINTS;
        } else if captured != EMPTY && owner_of(captured) != DEAD {
            info.points += PIECE_POINTS[type_of(captured)];
            if type_of(captured) == KING {
                let victim = owner_of(captured);
                info.eliminated = Some(victim);
                self.kill(victim);
            }
        }
        let after = kings_attacked_by(&self.board, color, self.eliminated);
        info.gives_check = after & !before != 0;
        if after & !before != 0 {
            info.check_bonus = match after.count_ones() {
                2 => 1,
                3 => 5,
                _ => 0,
            };
        }
        info.points += info.check_bonus;
        self.scores[color] += info.points;

        self.advance_turn();
        *self.counts.entry(self.hash).or_insert(0) += 1;
        self.undo.last_mut().unwrap().counted = true;
        info.points += self.award_end_bonus(color);
        Ok(info)
    }

    /// Resignation / timeout (chaturaji.game.eliminate). Undo with `pop`.
    pub fn eliminate(&mut self, color: usize) {
        let mut u = self.save();
        u.counted = false;
        self.undo.push(u);
        if !self.is_active(color) || self.over.is_some() {
            return;
        }
        self.kill(color);
        if self.current_player() == color {
            self.advance_turn();
        }
        self.award_end_bonus(usize::MAX);
    }

    pub fn pop(&mut self) -> Result<(), String> {
        let u = self.undo.pop().ok_or("pop from the root position")?;
        if u.counted {
            let n = self.counts.get_mut(&self.hash).expect("count of the current board");
            *n -= 1;
            if *n == 0 {
                self.counts.remove(&self.hash);
            }
        }
        self.board = u.board;
        self.turn = u.turn;
        self.scores = u.scores;
        self.eliminated = u.eliminated;
        self.hash = u.hash;
        self.over = u.over;
        Ok(())
    }

    /// The 16 global features from seat k's view (bot.mlp.global_features).
    pub fn globals(&self, k: usize) -> [f32; 16] {
        let mut g = [0f32; 16];
        let seats: [usize; 4] = [k % 4, (k + 1) % 4, (k + 2) % 4, (k + 3) % 4];
        let s: [f32; 4] = seats.map(|c| self.scores[c] as f32);
        let mean = (((s[0] + s[1]) + s[2]) + s[3]) / 4.0;
        for i in 0..4 {
            g[i] = if self.is_active(seats[i]) { 1.0 } else { 0.0 };
            g[4 + i] = s[i] / 30.0;
            g[8 + i] = (s[i] - mean) / 20.0;
        }
        g[12] = (self.repetitions() as f64 / 3.0) as f32;
        g[13] = (self.turn as f64 / 400.0).min(1.0) as f32;
        g[14] = (self.board.iter().filter(|&&p| p == code(DEAD, KING)).count() as f64 / 3.0) as f32;
        g[15] = (self.active_count() as f64 / 4.0) as f32;
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> usize {
        let b = name.as_bytes();
        (8 - (b[1] - b'0') as usize) * 8 + (b[0] - b'a') as usize
    }

    #[test]
    fn start_position() {
        let p = Position::initial();
        assert_eq!(p.board[sq("d1")], code(0, KING));
        assert_eq!(p.board[sq("h4")], code(3, KING));
        // Red: 4 pawn pushes + knight b1 -> c3 (a3 is free too) ... count must match Python (checked there)
        assert!(!p.legal_moves().is_empty());
        assert_eq!(p.repetitions(), 1);
    }

    #[test]
    fn push_pop_restores_everything() {
        let mut p = Position::initial();
        let start = p.clone();
        let mut seed = 1u64;
        for _ in 0..300 {
            if p.game_over().is_some() {
                break;
            }
            let ms = p.legal_moves();
            seed = splitmix(seed);
            p.push(ms[(seed % ms.len() as u64) as usize]).unwrap();
            assert_eq!(p.hash, board_hash(&p.board));
        }
        while p.depth() > 0 {
            p.pop().unwrap();
        }
        assert_eq!(p.board, start.board);
        assert_eq!(p.scores, start.scores);
        assert_eq!(p.turn, 0);
        assert_eq!(p.counts, start.counts);
        assert!(p.pop().is_err());
    }

    #[test]
    fn king_capture_eliminates() {
        let mut b = [EMPTY; 64];
        b[sq("d1")] = code(0, KING);
        b[sq("d4")] = code(0, ROOK);
        b[sq("d7")] = code(1, KING);
        b[sq("a5")] = code(1, PAWN);
        b[sq("h8")] = code(2, KING);
        b[sq("h1")] = code(3, KING);
        let mut p = Position::from_parts(b, 0, [0; 4], 0, vec![(b, 1)]).unwrap();
        let info = p.push(mv(sq("d4"), sq("d7"))).unwrap();
        assert_eq!(info.eliminated, Some(1));
        assert_eq!(p.board[sq("a5")], code(DEAD, PAWN));
        assert_eq!(p.current_player(), 2); // Blue is skipped
        assert!(info.points >= 3);
        p.pop().unwrap();
        assert_eq!(p.board, b);
        assert_eq!(p.eliminated, 0);
    }
}
