//! The greedy bot (bot/greedy.py, which stays the reference): a hand-written eval under
//! paranoid alpha-beta. Ported so the arena can run hundreds of games against it cheaply.
//! At a fixed depth it plays the same moves as the Python bot (`greedy_check` binary plus
//! `nn/check_greedy.py` compare them move by move over whole games).
//!
//! Every term follows bot/greedy.py line by line, including the order of float operations,
//! the move ordering (stable sorts over the same legal-move order) and the transposition
//! table (kept between moves, cleared when it holds more than half its size).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::rules::{
    code, from_sq, owner_of, pawn_capture_offsets, splitmix, to_sq, type_of, Move, Position, BISHOP, BISHOP_DIRS, DEAD,
    EMPTY, KING, KING_OFFSETS, KNIGHT, KNIGHT_OFFSETS, PAWN, PAWN_DIRECTION, PIECE_POINTS, ROOK, ROOK_DIRS,
};
use crate::search::RESIGN;

const PLACEMENT: [f64; 4] = [150.0, -20.0, -40.0, -100.0];
const THREAT_P: [f64; 3] = [0.9, 0.6, 0.4];
const INF: f64 = 1e9;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub loss_weight: f64,
    pub king_bonus: f64,
    pub king_bonus_rounds: u32,
    pub pawn_loss: f64,
    pub pawn_loss_rounds: u32,
    pub endgame_loss: f64,
    pub endgame_pawn_dist: i32,
    pub king_loss_rounds: u32,
    pub king_floor: f64,
    pub trade_bonus: f64,
    pub trade_rounds: u32,
    pub threat: bool,
    pub push_bonus: f64,
    pub push_from: u32,
    pub depth: u32,
    pub qdepth: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params { loss_weight: 1.2, king_bonus: 4.0, king_bonus_rounds: 10, pawn_loss: 2.0, pawn_loss_rounds: 15,
                 endgame_loss: 1.0, endgame_pawn_dist: 2, king_loss_rounds: 30, king_floor: 0.0, trade_bonus: 0.0,
                 trade_rounds: 1000, threat: true, push_bonus: 0.0, push_from: 15, depth: 4, qdepth: 4 }
    }
}

pub fn preset(name: &str) -> Option<Params> {
    let d = Params::default();
    match name {
        "greedy" => Some(d),
        "greedy-trade" => Some(Params { trade_bonus: 1.0, ..d }),
        "naive" => Some(Params { loss_weight: 1.0, king_bonus: 0.0, pawn_loss: 0.0, endgame_loss: 0.0, trade_bonus: 3.0,
                                 trade_rounds: 15, threat: false, push_bonus: 0.6, depth: 2, qdepth: 0, ..d }),
        _ => None,
    }
}

/// (endgame, trade): trade is (value I captured, plies since).
type Flags = (bool, Option<(i32, u32)>);

pub fn placement_value(scores: &[i32; 4], me: usize) -> f64 {
    let mine = scores[me];
    let better = (0..4).filter(|&c| c != me && scores[c] > mine).count();
    let tied = (0..4).filter(|&c| c != me && scores[c] == mine).count();
    let s: f64 = PLACEMENT[better..better + tied + 1].iter().sum();
    s / (tied as f64 + 1.0)
}

fn pawn_to_go(color: usize, r: i32, c: i32) -> i32 {
    let (dr, dc) = PAWN_DIRECTION[color];
    if dr != 0 {
        if dr > 0 { 7 - r } else { r }
    } else if dc > 0 {
        7 - c
    } else {
        c
    }
}

pub fn endgame_now(board: &[u8; 64]) -> bool {
    (0..64).any(|sq| {
        let p = board[sq];
        p != EMPTY && owner_of(p) != DEAD && type_of(p) == PAWN && pawn_to_go(owner_of(p), (sq / 8) as i32, (sq % 8) as i32) <= 2
    })
}

/// Value of the cheapest piece of `color` attacking (r, c); 0 if none.
pub fn attackers(board: &[u8; 64], r: i32, c: i32, color: usize) -> i32 {
    let at = |r: i32, c: i32| board[(r * 8 + c) as usize];
    let ok = |r: i32, c: i32| (0..8).contains(&r) && (0..8).contains(&c);
    let mut best = 99;
    for (odr, odc) in pawn_capture_offsets(color) {
        let (tr, tc) = (r - odr, c - odc);
        if ok(tr, tc) && at(tr, tc) == code(color, PAWN) {
            return 1;
        }
    }
    for (dr, dc) in KNIGHT_OFFSETS {
        let (tr, tc) = (r + dr, c + dc);
        if ok(tr, tc) && at(tr, tc) == code(color, KNIGHT) {
            best = 3;
        }
    }
    for (dr, dc) in KING_OFFSETS {
        let (tr, tc) = (r + dr, c + dc);
        if ok(tr, tc) && at(tr, tc) == code(color, KING) {
            best = 3;
        }
    }
    for (dirs, pt) in [(&ROOK_DIRS, ROOK), (&BISHOP_DIRS, BISHOP)] {
        for &(dr, dc) in dirs.iter() {
            let (mut tr, mut tc) = (r + dr, c + dc);
            while ok(tr, tc) {
                let t = at(tr, tc);
                if t != EMPTY {
                    if t == code(color, pt) && best > 5 {
                        best = 5;
                    }
                    break;
                }
                tr += dr;
                tc += dc;
            }
        }
    }
    if best == 99 { 0 } else { best }
}

fn active_kings(board: &[u8; 64]) -> i32 {
    board.iter().filter(|&&x| x != EMPTY && owner_of(x) != DEAD && type_of(x) == KING).count() as i32
}

#[derive(Clone, Copy)]
struct Entry {
    depth: i32,
    flag: u8,
    value: f64,
    best: Option<Move>,
}

#[derive(Clone, Debug, Default)]
pub struct GreedyResult {
    pub best: Move,
    pub value: f64,
    pub depth: u32,
    pub nodes: u64,
    pub seconds: f64,
}

pub struct Greedy {
    pub p: Params,
    pub depth: Option<u32>,
    pub time_ms: Option<f64>,
    pub max_depth: u32,
    pub tt_size: usize,
    tt: HashMap<u64, Entry>,
    nodes: u64,
    deadline: Option<Instant>,
    aborted: bool,
}

impl Greedy {
    pub fn new(p: Params, depth: Option<u32>, time_ms: Option<f64>) -> Self {
        Greedy { p, depth, time_ms, max_depth: 12, tt_size: 400_000, tt: HashMap::new(), nodes: 0, deadline: None, aborted: false }
    }

    /// (value change for me from this move, new flags). Needs the parent's turn, board
    /// (king count) and the moving piece; `info` is the move's result.
    fn delta(&self, me: usize, turn: u32, parent_kings: i32, mover: usize, piece: u8, to: usize,
             info: &crate::rules::MoveInfo, flags: Flags) -> (f64, Flags) {
        let p = &self.p;
        let r = turn / 4;
        let (mut endgame, mut trade) = flags;
        let mut d = 0.0f64;
        if let Some((v, n)) = trade {
            trade = if n < 3 { Some((v, n + 1)) } else { None };
        }
        let cap = info.captured;
        if mover == me {
            d += info.points as f64;
            if cap != EMPTY && owner_of(cap) != DEAD {
                if type_of(cap) == KING && r < p.king_bonus_rounds {
                    d += p.king_bonus;
                }
                if owner_of(cap) != me {
                    trade = Some((PIECE_POINTS[type_of(cap)], 0));
                }
            }
            if p.push_bonus != 0.0 && r >= p.push_from && type_of(piece) == PAWN {
                d += p.push_bonus;
            }
        } else if cap != EMPTY && owner_of(cap) == me {
            let v = PIECE_POINTS[type_of(cap)];
            if type_of(cap) == KING {
                if r < p.king_loss_rounds {
                    d -= ((p.king_loss_rounds - r) as i32 * parent_kings) as f64;
                }
                d -= p.king_floor;
                d -= p.loss_weight * v as f64;
            } else {
                d -= p.loss_weight * v as f64;
                if type_of(cap) == PAWN && r < p.pawn_loss_rounds {
                    d -= p.pawn_loss;
                }
                if endgame {
                    d -= p.endgame_loss;
                }
                if let Some((tv, _)) = trade {
                    if v <= tv && r < p.trade_rounds {
                        d += p.trade_bonus;
                        trade = None;
                    }
                }
            }
        }
        if !endgame {
            if info.eliminated.is_some() && r >= 10 {
                endgame = true;
            } else if type_of(piece) == PAWN {
                if pawn_to_go(mover, (to / 8) as i32, (to % 8) as i32) <= p.endgame_pawn_dist {
                    endgame = true;
                }
            }
        }
        (d, (endgame, trade))
    }

    /// Expected loss at a leaf from my pieces that opponents attack right now.
    fn threat(&self, me: usize, pos: &Position) -> f64 {
        let board = &pos.board;
        let r = pos.turn / 4;
        let mut order: Vec<usize> = Vec::with_capacity(3);
        let s = pos.current_player();
        for k in 0..4 {
            let c = (s + k) % 4;
            if c == me {
                break;
            }
            if pos.is_active(c) {
                order.push(c);
            }
        }
        let mine_to_move = order.is_empty();
        if mine_to_move {
            for k in 1..4 {
                let c = (me + k) % 4;
                if pos.is_active(c) {
                    order.push(c);
                }
            }
        }
        let mut kings: Option<i32> = None;
        let mut losses: Vec<f64> = Vec::new();
        for sq in 0..64 {
            let pc = board[sq];
            if pc == EMPTY || owner_of(pc) != me {
                continue;
            }
            let (rr, cc) = ((sq / 8) as i32, (sq % 8) as i32);
            let v = PIECE_POINTS[type_of(pc)];
            let mut worst = 0.0f64;
            for (k, &c) in order.iter().enumerate() {
                let a = attackers(board, rr, cc, c);
                if a == 0 {
                    continue;
                }
                let loss = if type_of(pc) == KING {
                    let kg = *kings.get_or_insert_with(|| active_kings(board));
                    ((self.p.king_loss_rounds as i32 - r as i32).max(0) * kg) as f64 + self.p.loss_weight * v as f64
                } else {
                    let defended = attackers(board, rr, cc, me) > 0;
                    let net = if defended { ((v - a) as f64).max(0.0) } else { v as f64 };
                    self.p.loss_weight * net
                };
                let idx = (k + if mine_to_move { 1 } else { 0 }).min(2);
                worst = worst.max(loss * THREAT_P[idx]);
            }
            if worst != 0.0 {
                losses.push(worst);
            }
        }
        if losses.is_empty() {
            return 0.0;
        }
        if mine_to_move {
            let mut mi = 0;
            for i in 1..losses.len() {
                if losses[i] > losses[mi] {
                    mi = i;
                }
            }
            losses.remove(mi);
        }
        let mut sum = 0.0;
        for x in losses {
            sum += x;
        }
        sum
    }

    fn leaf(&self, me: usize, pos: &Position) -> f64 {
        if self.p.threat { -self.threat(me, pos) } else { 0.0 }
    }

    fn claim(&self, me: usize, pos: &mut Position) -> Option<f64> {
        if pos.active_count() != 2 {
            return None;
        }
        let cur = pos.current_player();
        pos.eliminate(cur);
        let v = placement_value(&pos.scores, me);
        pos.pop().expect("pop");
        Some(v)
    }

    fn key(pos: &Position, flags: Flags) -> u64 {
        let mut h = splitmix((pos.turn % 4) as u64 | (((pos.turn / 4).min(30) as u64) << 2) | ((pos.eliminated as u64) << 8));
        for s in pos.scores {
            h = splitmix(h ^ (s as u32 as u64));
        }
        let f = match flags.1 {
            None => 0u64,
            Some((v, n)) => 1 + ((v as u64) << 1) + ((n as u64) << 8),
        };
        h = splitmix(h ^ (flags.0 as u64) ^ (f << 1));
        pos.hash ^ h
    }

    fn tick(&mut self) {
        self.nodes += 1;
        if let Some(d) = self.deadline {
            if self.nodes & 127 == 0 && Instant::now() > d {
                self.aborted = true;
            }
        }
    }

    fn ordered(pos: &Position, moves: &[Move], tt_move: Option<Move>, only_captures: bool) -> Vec<Move> {
        let board = &pos.board;
        let mut scored: Vec<(i32, Move)> = Vec::with_capacity(moves.len());
        for &m in moves {
            let mut victim = board[to_sq(m)];
            let mut gain = 0;
            if victim != EMPTY {
                gain = if victim == code(DEAD, KING) { 3 } else if owner_of(victim) == DEAD { 0 } else { PIECE_POINTS[type_of(victim)] };
                if owner_of(victim) == DEAD && type_of(victim) != KING {
                    victim = EMPTY;
                }
            }
            if only_captures && (victim == EMPTY || gain == 0) {
                continue;
            }
            let att = PIECE_POINTS[type_of(board[from_sq(m)])];
            let key = if tt_move == Some(m) { 1000 } else if victim != EMPTY { gain * 10 - att + 100 } else { 0 };
            scored.push((key, m));
        }
        scored.sort_by_key(|t| -t.0);
        scored.into_iter().map(|t| t.1).collect()
    }

    /// Play `m`: returns (delta, new flags, terminal) with the move made on `pos`
    /// (the caller pops). Terminal values include the placement.
    fn child(&self, me: usize, pos: &mut Position, m: Move, flags: Flags) -> (f64, Option<Flags>, bool) {
        let turn = pos.turn;
        let mover = pos.current_player();
        let piece = pos.board[from_sq(m)];
        let cap = pos.board[to_sq(m)];
        let parent_kings = if cap != EMPTY && owner_of(cap) == me && type_of(cap) == KING { active_kings(&pos.board) } else { 0 };
        let info = pos.push(m).expect("legal move");
        let (d, nf) = self.delta(me, turn, parent_kings, mover, piece, to_sq(m), &info, flags);
        if !pos.is_active(me) {
            return (d, None, true);
        }
        if pos.game_over().is_some() {
            return (d + placement_value(&pos.scores, me), None, true);
        }
        (d, Some(nf), false)
    }

    fn quiesce(&mut self, me: usize, pos: &mut Position, mut alpha: f64, mut beta: f64, flags: Flags, q: u32) -> f64 {
        self.tick();
        if self.aborted {
            return 0.0;
        }
        let maxing = pos.current_player() == me;
        let mut stand = self.leaf(me, pos);
        if let Some(claim) = self.claim(me, pos) {
            stand = if maxing { stand.max(claim) } else { stand.min(claim) };
        }
        if q == 0 {
            return stand;
        }
        if maxing {
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
        } else {
            if stand <= alpha {
                return stand;
            }
            beta = beta.min(stand);
        }
        let mut best = stand;
        let moves = pos.legal_moves();
        for m in Self::ordered(pos, &moves, None, true) {
            let (d, nf, term) = self.child(me, pos, m, flags);
            let v = if term { d } else { d + self.quiesce(me, pos, alpha - d, beta - d, nf.unwrap(), q - 1) };
            pos.pop().expect("pop");
            if self.aborted {
                return 0.0;
            }
            if maxing {
                if v > best {
                    best = v;
                    alpha = alpha.max(v);
                }
            } else if v < best {
                best = v;
                beta = beta.min(v);
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    fn search(&mut self, me: usize, pos: &mut Position, depth: i32, mut alpha: f64, mut beta: f64, flags: Flags) -> f64 {
        if depth <= 0 {
            return self.quiesce(me, pos, alpha, beta, flags, self.p.qdepth);
        }
        self.tick();
        if self.aborted {
            return 0.0;
        }
        let key = Self::key(pos, flags);
        let mut tt_move = None;
        let (a0, b0) = (alpha, beta);
        if let Some(e) = self.tt.get(&key).copied() {
            tt_move = e.best;
            if e.depth >= depth {
                if e.flag == 0 {
                    return e.value;
                }
                if e.flag == 1 {
                    alpha = alpha.max(e.value);
                } else {
                    beta = beta.min(e.value);
                }
                if alpha >= beta {
                    return e.value;
                }
            }
        }
        let maxing = pos.current_player() == me;
        let mut best = if maxing { -INF } else { INF };
        let mut best_move = None;
        if let Some(claim) = self.claim(me, pos) {
            best = claim;
            if maxing {
                alpha = alpha.max(best);
            } else {
                beta = beta.min(best);
            }
        }
        if alpha < beta {
            let moves = pos.legal_moves();
            for m in Self::ordered(pos, &moves, tt_move, false) {
                let (d, nf, term) = self.child(me, pos, m, flags);
                let v = if term { d } else { d + self.search(me, pos, depth - 1, alpha - d, beta - d, nf.unwrap()) };
                pos.pop().expect("pop");
                if self.aborted {
                    return 0.0;
                }
                if maxing {
                    if v > best {
                        best = v;
                        best_move = Some(m);
                        alpha = alpha.max(v);
                    }
                } else if v < best {
                    best = v;
                    best_move = Some(m);
                    beta = beta.min(v);
                }
                if alpha >= beta {
                    break;
                }
            }
        }
        let flag = if a0 < best && best < b0 { 0 } else if best >= b0 { 1 } else { 2 };
        if self.tt.len() < self.tt_size {
            self.tt.insert(key, Entry { depth, flag, value: best, best: best_move });
        }
        best
    }

    /// The move for the side to move (RESIGN = claim the win).
    pub fn best_move(&mut self, pos: &mut Position) -> GreedyResult {
        let t0 = Instant::now();
        let me = pos.current_player();
        let moves = pos.legal_moves();
        assert!(!moves.is_empty(), "no legal moves");
        let flags0: Flags = (endgame_now(&pos.board), None);
        self.nodes = 0;
        self.aborted = false;
        self.deadline = match (self.time_ms, self.depth) {
            (Some(ms), None) => Some(t0 + Duration::from_secs_f64(ms / 1000.0)),
            _ => None,
        };
        if self.tt.len() > self.tt_size / 2 {
            self.tt.clear();
        }
        let claim = self.claim(me, pos);
        let (mut best, mut bv, mut reached) = (moves[0], -INF, 0);
        let limit = match (self.depth, self.time_ms) {
            (Some(d), _) => d,
            (None, Some(_)) => self.max_depth,
            (None, None) => self.p.depth,
        };
        let mut order = Self::ordered(pos, &moves, None, false);
        for d in 1..=limit as i32 {
            let mut alpha = -INF;
            let (mut cur_best, mut cur_v) = (None, -INF);
            let mut scores: Vec<(f64, Move)> = Vec::with_capacity(order.len());
            for &m in &order {
                let (dl, nf, term) = self.child(me, pos, m, flags0);
                let v = if term { dl } else { dl + self.search(me, pos, d - 1, alpha - dl, INF, nf.unwrap()) };
                pos.pop().expect("pop");
                if self.aborted {
                    break;
                }
                scores.push((v, m));
                if v > cur_v {
                    cur_v = v;
                    cur_best = Some(m);
                    alpha = alpha.max(v);
                }
            }
            if self.aborted {
                break;
            }
            best = cur_best.unwrap();
            bv = cur_v;
            reached = d as u32;
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            order = scores.into_iter().map(|t| t.1).collect();
        }
        if let Some(c) = claim {
            if c > bv {
                best = RESIGN;
                bv = c;
            }
        }
        GreedyResult { best, value: bv, depth: reached, nodes: self.nodes, seconds: t0.elapsed().as_secs_f64() }
    }
}
