//! Greedy bot (bot/greedy.py) in Rust, idea G15, plus switches for the G-ideas.
//!
//! With the default `GreedyParams` (all new switches off) it is the Python bot: same move
//! ordering, same "value to go" eval, same transposition-table policy, so at a fixed depth it
//! plays the same moves with the same values and node counts (`bin/greedy_check.rs` replays the
//! Python reference games of `tools/greedy_ref.py`).
//!
//! New switches (all default off / 0):
//! - `one_capture` (G05): in the threat penalty each opponent takes at most one of my pieces,
//!   its biggest, and two opponents can't take the same piece.
//! - `king_safety` (G07): leaf term -w * (squares around my king attacked by opponents),
//!   scaled by how much my king is still worth ((king_loss_rounds - round) / king_loss_rounds,
//!   at least 0.25).
//! - `king_floor` (G09): already in Python; extra penalty for losing my king at any round.
//! - `offense` (G06): leaf term +w * sum over opponents of the second-biggest undefended piece
//!   of that opponent that I attack (a fork wins the smaller piece).
//! - `material` (G10): leaf term +w * (my live material - my material at the root).
//! - `tiebreak` (G14): leaf term +w * (table v2 strength of me - mean of the others), needs `pst2`.
//! - `placement=0`: no placement term at game end (raw points; a G02 ablation).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::pst2::Pst2;
use crate::rules::{code, from_sq, owner_of, to_sq, type_of, Move, Position, DEAD, EMPTY, KING, PAWN};

const PIECE_POINTS: [i32; 5] = [3, 5, 3, 5, 1];
const PAWN_DIR: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
const KNIGHT_OFF: [(i32, i32); 8] = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
const KING_OFF: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
const ROOK_DIRS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const INF: f64 = 1e9;

/// The mover leaves the game (claims the win / resigns). Not a board move.
pub const RESIGN: Move = 0xFFFE;
const NO_MOVE: Move = 0xFFFF;

#[derive(Clone, Debug)]
pub struct GreedyParams {
    pub loss_weight: f64,
    pub king_bonus: f64,
    pub king_bonus_rounds: i32,
    pub pawn_loss: f64,
    pub pawn_loss_rounds: i32,
    pub endgame_loss: f64,
    pub endgame_pawn_dist: i32,
    pub king_loss_rounds: i32,
    pub king_floor: f64,
    pub trade_bonus: f64,
    pub trade_rounds: i32,
    pub threat: bool,
    pub push_bonus: f64,
    pub push_from: i32,
    pub depth: u32,
    pub qdepth: u32,
    pub threat_p: [f64; 3],
    pub placement: [f64; 4],
    pub use_placement: bool,
    // new switches
    pub one_capture: bool,
    pub king_safety: f64,
    pub offense: f64,
    pub material: f64,
    pub tiebreak: f64,
    pub pst2: Option<Arc<Pst2>>,
}

impl Default for GreedyParams {
    fn default() -> Self {
        GreedyParams {
            loss_weight: 1.2, king_bonus: 4.0, king_bonus_rounds: 10, pawn_loss: 2.0, pawn_loss_rounds: 15,
            endgame_loss: 1.0, endgame_pawn_dist: 2, king_loss_rounds: 30, king_floor: 0.0, trade_bonus: 0.0,
            trade_rounds: 1000, threat: true, push_bonus: 0.0, push_from: 15, depth: 4, qdepth: 4,
            threat_p: [0.9, 0.6, 0.4], placement: [150.0, -20.0, -40.0, -100.0], use_placement: true,
            one_capture: false, king_safety: 0.0, offense: 0.0, material: 0.0, tiebreak: 0.0, pst2: None,
        }
    }
}

impl GreedyParams {
    pub fn preset(name: &str) -> Result<GreedyParams, String> {
        let mut p = GreedyParams::default();
        match name {
            "greedy" => {}
            "greedy-trade" => p.trade_bonus = 1.0,
            "naive" => {
                p.loss_weight = 1.0; p.king_bonus = 0.0; p.pawn_loss = 0.0; p.endgame_loss = 0.0;
                p.trade_bonus = 3.0; p.trade_rounds = 15; p.threat = false; p.push_bonus = 0.6; p.depth = 2; p.qdepth = 0;
            }
            _ => return Err(format!("unknown greedy preset {}", name)),
        }
        Ok(p)
    }

    /// Set one key (the arena spec keys `g_<name>`).
    pub fn set(&mut self, k: &str, v: &str) -> Result<(), String> {
        let f = || v.parse::<f64>().map_err(|e| format!("greedy {}: {}", k, e));
        let i = || v.parse::<i32>().map_err(|e| format!("greedy {}: {}", k, e));
        let b = || match v { "0" => Ok(false), "1" => Ok(true), _ => Err(format!("greedy {}: want 0 or 1", k)) };
        match k {
            "loss_weight" => self.loss_weight = f()?,
            "king_bonus" => self.king_bonus = f()?,
            "king_bonus_rounds" => self.king_bonus_rounds = i()?,
            "pawn_loss" => self.pawn_loss = f()?,
            "pawn_loss_rounds" => self.pawn_loss_rounds = i()?,
            "endgame_loss" => self.endgame_loss = f()?,
            "king_loss_rounds" => self.king_loss_rounds = i()?,
            "king_floor" => self.king_floor = f()?,
            "trade_bonus" => self.trade_bonus = f()?,
            "threat" => self.threat = b()?,
            "push_bonus" => self.push_bonus = f()?,
            "qdepth" => self.qdepth = i()? as u32,
            "tp1" => self.threat_p[0] = f()?,
            "tp2" => self.threat_p[1] = f()?,
            "tp3" => self.threat_p[2] = f()?,
            "placement" => self.use_placement = b()?,
            "one_capture" => self.one_capture = b()?,
            "king_safety" => self.king_safety = f()?,
            "offense" => self.offense = f()?,
            "material" => self.material = f()?,
            "tiebreak" => self.tiebreak = f()?,
            _ => return Err(format!("unknown greedy key {}", k)),
        }
        Ok(())
    }
}

#[inline]
fn on_board(r: i32, c: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&c)
}

fn pawn_capture_offsets(color: usize) -> [(i32, i32); 2] {
    let (dr, dc) = PAWN_DIR[color];
    if dr != 0 { [(dr, -1), (dr, 1)] } else { [(-1, dc), (1, dc)] }
}

/// Value of the cheapest piece of `color` attacking `sq` (P 1, N/K 3, R/B 5); 0 if none.
pub fn cheapest_attacker(b: &[u8; 64], sq: usize, color: usize) -> i32 {
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    for (odr, odc) in pawn_capture_offsets(color) {
        let (tr, tc) = (r - odr, c - odc);
        if on_board(tr, tc) && b[(tr * 8 + tc) as usize] == code(color, PAWN) {
            return 1;
        }
    }
    for &(dr, dc) in &KNIGHT_OFF {
        if on_board(r + dr, c + dc) && b[((r + dr) * 8 + c + dc) as usize] == code(color, 2) {
            return 3;
        }
    }
    for &(dr, dc) in &KING_OFF {
        if on_board(r + dr, c + dc) && b[((r + dr) * 8 + c + dc) as usize] == code(color, KING) {
            return 3;
        }
    }
    for (dirs, ty) in [(&ROOK_DIRS, 3usize), (&BISHOP_DIRS, 1usize)] {
        for &(dr, dc) in dirs {
            let (mut tr, mut tc) = (r + dr, c + dc);
            while on_board(tr, tc) {
                let t = b[(tr * 8 + tc) as usize];
                if t != EMPTY {
                    if t == code(color, ty) {
                        return 5;
                    }
                    break;
                }
                tr += dr;
                tc += dc;
            }
        }
    }
    0
}

pub fn pawn_to_go(color: usize, sq: usize) -> i32 {
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    let (dr, dc) = PAWN_DIR[color];
    if dr != 0 { if dr > 0 { 7 - r } else { r } } else if dc > 0 { 7 - c } else { c }
}

pub fn endgame_now(b: &[u8; 64]) -> bool {
    (0..64).any(|s| {
        let p = b[s];
        p != EMPTY && owner_of(p) != DEAD && type_of(p) == PAWN && pawn_to_go(owner_of(p), s) <= 2
    })
}

pub fn placement_value(sc: &[i32; 4], me: usize, placement: &[f64; 4]) -> f64 {
    let (mut better, mut tied) = (0usize, 0usize);
    for c in 0..4 {
        if c != me {
            better += (sc[c] > sc[me]) as usize;
            tied += (sc[c] == sc[me]) as usize;
        }
    }
    placement[better..=better + tied].iter().sum::<f64>() / (tied + 1) as f64
}

fn live_kings(b: &[u8; 64]) -> i32 {
    b.iter().filter(|&&p| p != EMPTY && owner_of(p) != DEAD && type_of(p) == KING).count() as i32
}

fn material(b: &[u8; 64], me: usize) -> i32 {
    b.iter().filter(|&&p| p != EMPTY && owner_of(p) == me && type_of(p) != KING).map(|&p| PIECE_POINTS[type_of(p)]).sum()
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Flags {
    endgame: bool,
    trade_v: i32, // -1 = none
    trade_plies: i32,
}

impl Flags {
    fn key(&self) -> u64 {
        self.endgame as u64 | (((self.trade_v + 1) as u64) << 1) | ((self.trade_plies as u64) << 8)
    }
}

#[derive(Clone, Copy)]
struct Tte {
    depth: u32,
    flag: u8,
    v: f64,
    m: Move,
}

struct Stop;

#[derive(Clone, Debug, Default)]
pub struct GreedyResult {
    pub best: Move,
    pub depth: u32,
    pub nodes: u64,
    pub value: f64,
    pub seconds: f64,
}

pub struct GreedyBot {
    pub p: GreedyParams,
    pub fixed_depth: Option<u32>,
    pub time_ms: Option<f64>,
    pub node_limit: Option<u64>,
    pub max_depth: u32,
    pub tt_size: usize,
    tt: HashMap<u64, Tte>,
    me: usize,
    nodes: u64,
    deadline: Option<Instant>,
    root_material: i32,
    buf: Vec<Vec<Move>>,
}

impl GreedyBot {
    pub fn new(p: GreedyParams) -> Self {
        GreedyBot { p, fixed_depth: None, time_ms: None, node_limit: None, max_depth: 12, tt_size: 400_000,
                    tt: HashMap::with_capacity(1 << 16), me: 0, nodes: 0, deadline: None, root_material: 0,
                    buf: Vec::new() }
    }

    fn tick(&mut self) -> Result<(), Stop> {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() > d {
                    return Err(Stop);
                }
            }
        }
        if let Some(n) = self.node_limit {
            if self.nodes > n {
                return Err(Stop);
            }
        }
        Ok(())
    }

    fn key(pos: &Position, f: &Flags) -> u64 {
        use crate::rules::splitmix;
        let mut h = pos.hash;
        h ^= splitmix(0x1234 + (pos.turn % 4) as u64);
        h ^= splitmix(0x5678 + (pos.turn / 4).min(30) as u64);
        for i in 0..4 {
            h ^= splitmix(0x9abc + i as u64 * 1000003 + (pos.scores[i] + 500) as u64);
        }
        h ^= splitmix(0xdef0 + pos.eliminated as u64);
        h ^ splitmix(0x4242 + f.key())
    }

    /// Opponents in the order they move before my next turn, and whether I am to move.
    fn movers_before_me(&self, pos: &Position) -> ([usize; 3], usize, bool) {
        let (mut order, mut no) = ([0usize; 3], 0usize);
        let s = pos.current_player();
        for k in 0..4 {
            let c = (s + k) % 4;
            if c == self.me {
                break;
            }
            if pos.is_active(c) {
                order[no] = c;
                no += 1;
            }
        }
        if no > 0 {
            return (order, no, false);
        }
        for k in 1..4 {
            let c = (self.me + k) % 4;
            if pos.is_active(c) {
                order[no] = c;
                no += 1;
            }
        }
        (order, no, true)
    }

    /// Greedy's threat penalty (bot/greedy.py `_threat`), or its G05 variant.
    pub fn threat(&self, pos: &Position) -> f64 {
        let b = &pos.board;
        let r = (pos.turn / 4) as i32;
        let (order, no, mine_to_move) = self.movers_before_me(pos);
        let mut kings = -1;
        // per piece: loss if attacker k takes it (already weighted), for the G05 variant
        let mut table: Vec<[f64; 3]> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        for sq in 0..64 {
            let pc = b[sq];
            if pc == EMPTY || owner_of(pc) != self.me {
                continue;
            }
            let t = type_of(pc);
            let v = PIECE_POINTS[t] as f64;
            let mut worst = 0.0f64;
            let mut row = [0.0f64; 3];
            for k in 0..no {
                let a = cheapest_attacker(b, sq, order[k]);
                if a == 0 {
                    continue;
                }
                let loss = if t == KING {
                    if kings < 0 {
                        kings = live_kings(b);
                    }
                    (self.p.king_loss_rounds - r).max(0) as f64 * kings as f64 + self.p.loss_weight * v
                } else {
                    let defended = cheapest_attacker(b, sq, self.me) > 0;
                    let net = if defended { (v - a as f64).max(0.0) } else { v };
                    self.p.loss_weight * net
                };
                let w = loss * self.p.threat_p[(k + mine_to_move as usize).min(2)];
                row[k] = w;
                worst = worst.max(w);
            }
            if worst != 0.0 {
                losses.push(worst);
                table.push(row);
            }
        }
        if losses.is_empty() {
            return 0.0;
        }
        if !self.p.one_capture {
            let mut skip = usize::MAX;
            if mine_to_move {
                skip = 0;
                for i in 1..losses.len() {
                    if losses[i] > losses[skip] {
                        skip = i;
                    }
                }
            }
            return losses.iter().enumerate().filter(|(i, _)| *i != skip).map(|(_, x)| x).sum();
        }
        // G05: I save my most threatened piece if I move first; then each opponent in turn
        // order takes its best remaining piece.
        let mut taken = vec![false; table.len()];
        if mine_to_move {
            let mut skip = 0;
            for i in 1..losses.len() {
                if losses[i] > losses[skip] {
                    skip = i;
                }
            }
            taken[skip] = true;
        }
        let mut total = 0.0;
        for k in 0..no {
            let mut best = (0.0, usize::MAX);
            for (i, row) in table.iter().enumerate() {
                if !taken[i] && row[k] > best.0 {
                    best = (row[k], i);
                }
            }
            if best.1 != usize::MAX {
                taken[best.1] = true;
                total += best.0;
            }
        }
        total
    }

    fn king_safety(&self, pos: &Position) -> f64 {
        let b = &pos.board;
        let Some(ks) = b.iter().position(|&p| p == code(self.me, KING)) else { return 0.0 };
        let (r, c) = ((ks / 8) as i32, (ks % 8) as i32);
        let mut n = 0;
        for dr in -1..=1 {
            for dc in -1..=1 {
                if !on_board(r + dr, c + dc) {
                    continue;
                }
                let sq = ((r + dr) * 8 + c + dc) as usize;
                for o in 0..4 {
                    if o != self.me && pos.is_active(o) && cheapest_attacker(b, sq, o) > 0 {
                        n += 1;
                    }
                }
            }
        }
        let round = (pos.turn / 4) as i32;
        let scale = ((self.p.king_loss_rounds - round) as f64 / self.p.king_loss_rounds as f64).max(0.25);
        -self.p.king_safety * n as f64 * scale
    }

    fn offense(&self, pos: &Position) -> f64 {
        let b = &pos.board;
        let mut total = 0.0;
        for o in 0..4 {
            if o == self.me || !pos.is_active(o) {
                continue;
            }
            let (mut a1, mut a2) = (0i32, 0i32);
            for sq in 0..64 {
                let p = b[sq];
                if p == EMPTY || owner_of(p) != o || type_of(p) == KING {
                    continue;
                }
                if cheapest_attacker(b, sq, self.me) > 0 && cheapest_attacker(b, sq, o) == 0 {
                    let v = PIECE_POINTS[type_of(p)];
                    if v > a1 {
                        a2 = a1;
                        a1 = v;
                    } else if v > a2 {
                        a2 = v;
                    }
                }
            }
            total += a2 as f64;
        }
        self.p.offense * total
    }

    fn leaf(&self, pos: &Position) -> f64 {
        let mut v = if self.p.threat { -self.threat(pos) } else { 0.0 };
        if self.p.king_safety != 0.0 {
            v += self.king_safety(pos);
        }
        if self.p.offense != 0.0 {
            v += self.offense(pos);
        }
        if self.p.material != 0.0 {
            v += self.p.material * (material(&pos.board, self.me) - self.root_material) as f64;
        }
        if self.p.tiebreak != 0.0 {
            if let Some(t) = &self.p.pst2 {
                let s = t.seat_scores(pos);
                let others: f32 = (0..4).filter(|&o| o != self.me).map(|o| s[o]).sum::<f32>() / 3.0;
                v += self.p.tiebreak * (s[self.me] - others) as f64;
            }
        }
        v
    }

    fn claim(&self, pos: &mut Position) -> Option<f64> {
        if pos.eliminated.count_ones() != 2 {
            return None;
        }
        pos.eliminate(pos.current_player());
        let v = placement_value(&pos.scores, self.me, &self.p.placement);
        pos.pop().unwrap();
        Some(v)
    }

    fn ordered(&self, pos: &Position, moves: &[Move], tt_move: Move, only_captures: bool, out: &mut Vec<Move>) {
        let mut sc: Vec<(i32, Move)> = Vec::with_capacity(moves.len());
        for &m in moves {
            let victim = pos.board[to_sq(m)];
            let mut gain = 0;
            let mut has = victim != EMPTY;
            if has {
                gain = if victim == code(DEAD, KING) { 3 } else if owner_of(victim) == DEAD { 0 } else { PIECE_POINTS[type_of(victim)] };
                if owner_of(victim) == DEAD && type_of(victim) != KING {
                    has = false;
                }
            }
            if only_captures && (!has || gain == 0) {
                continue;
            }
            let att = PIECE_POINTS[type_of(pos.board[from_sq(m)])];
            let k = if m == tt_move { 1000 } else if has { gain * 10 - att + 100 } else { 0 };
            sc.push((k, m));
        }
        sc.sort_by(|a, b| b.0.cmp(&a.0)); // stable
        out.clear();
        out.extend(sc.iter().map(|x| x.1));
    }

    /// Push m; returns (value change for me, new flags, line over).
    fn child(&self, pos: &mut Position, m: Move, flags: &Flags) -> (f64, Flags, bool) {
        let p = &self.p;
        let me = self.me;
        let r = (pos.turn / 4) as i32;
        let cap = pos.board[to_sq(m)];
        let piece = pos.board[from_sq(m)];
        let kings_before = if cap != EMPTY && owner_of(cap) == me && type_of(cap) == KING { live_kings(&pos.board) } else { 0 };
        let info = pos.push(m).expect("legal move");
        let mover = owner_of(piece);
        let mut nf = *flags;
        let mut d = 0.0;
        if nf.trade_v >= 0 {
            if nf.trade_plies < 3 {
                nf.trade_plies += 1;
            } else {
                nf.trade_v = -1;
                nf.trade_plies = 0;
            }
        }
        if mover == me {
            d += info.points as f64;
            if cap != EMPTY && owner_of(cap) != DEAD {
                if type_of(cap) == KING && r < p.king_bonus_rounds {
                    d += p.king_bonus;
                }
                nf.trade_v = PIECE_POINTS[type_of(cap)];
                nf.trade_plies = 0;
            }
            if p.push_bonus != 0.0 && r >= p.push_from && type_of(piece) == PAWN {
                d += p.push_bonus;
            }
        } else if cap != EMPTY && owner_of(cap) == me {
            let v = PIECE_POINTS[type_of(cap)] as f64;
            if type_of(cap) == KING {
                if r < p.king_loss_rounds {
                    d -= (p.king_loss_rounds - r) as f64 * kings_before as f64;
                }
                d -= p.king_floor;
                d -= p.loss_weight * v;
            } else {
                d -= p.loss_weight * v;
                if type_of(cap) == PAWN && r < p.pawn_loss_rounds {
                    d -= p.pawn_loss;
                }
                if flags.endgame {
                    d -= p.endgame_loss;
                }
                if nf.trade_v >= 0 && v <= nf.trade_v as f64 && r < p.trade_rounds {
                    d += p.trade_bonus;
                    nf.trade_v = -1;
                    nf.trade_plies = 0;
                }
            }
        }
        if !flags.endgame {
            if info.eliminated.is_some() && r >= 10 {
                nf.endgame = true;
            } else if type_of(piece) == PAWN && pawn_to_go(mover, to_sq(m)) <= p.endgame_pawn_dist {
                nf.endgame = true;
            }
        }
        if !pos.is_active(me) {
            return (d, nf, true);
        }
        if pos.game_over().is_some() {
            if p.use_placement {
                d += placement_value(&pos.scores, me, &p.placement);
            }
            return (d, nf, true);
        }
        (d, nf, false)
    }

    fn take_buf(&mut self) -> Vec<Move> {
        self.buf.pop().unwrap_or_else(|| Vec::with_capacity(64))
    }

    fn quiesce(&mut self, pos: &mut Position, mut alpha: f64, mut beta: f64, flags: &Flags, q: u32) -> Result<f64, Stop> {
        self.tick()?;
        let maxing = pos.current_player() == self.me;
        let mut stand = self.leaf(pos);
        if let Some(c) = self.claim(pos) {
            stand = if maxing { stand.max(c) } else { stand.min(c) };
        }
        if q == 0 {
            return Ok(stand);
        }
        if maxing {
            if stand >= beta {
                return Ok(stand);
            }
            alpha = alpha.max(stand);
        } else {
            if stand <= alpha {
                return Ok(stand);
            }
            beta = beta.min(stand);
        }
        let mut best = stand;
        let mut legal = self.take_buf();
        pos.legal_moves_into(&mut legal);
        let mut order = self.take_buf();
        self.ordered(pos, &legal, NO_MOVE, true, &mut order);
        let mut res = Ok(());
        for &m in &order {
            let (d, nf, term) = self.child(pos, m, flags);
            let v = if term {
                d
            } else {
                match self.quiesce(pos, alpha - d, beta - d, &nf, q - 1) {
                    Ok(x) => d + x,
                    Err(e) => {
                        pos.pop().unwrap();
                        res = Err(e);
                        break;
                    }
                }
            };
            pos.pop().unwrap();
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
        self.buf.push(legal);
        self.buf.push(order);
        res.map(|_| best)
    }

    fn search(&mut self, pos: &mut Position, depth: u32, mut alpha: f64, mut beta: f64, flags: &Flags) -> Result<f64, Stop> {
        if depth == 0 {
            return self.quiesce(pos, alpha, beta, flags, self.p.qdepth);
        }
        self.tick()?;
        let k = Self::key(pos, flags);
        let mut tt_move = NO_MOVE;
        let (a0, b0) = (alpha, beta);
        if let Some(e) = self.tt.get(&k) {
            tt_move = e.m;
            if e.depth >= depth {
                if e.flag == 0 {
                    return Ok(e.v);
                }
                if e.flag == 1 {
                    alpha = alpha.max(e.v);
                } else {
                    beta = beta.min(e.v);
                }
                if alpha >= beta {
                    return Ok(e.v);
                }
            }
        }
        let maxing = pos.current_player() == self.me;
        let mut best = if maxing { -INF } else { INF };
        let mut best_move = NO_MOVE;
        if let Some(c) = self.claim(pos) {
            best = c;
            if maxing {
                alpha = alpha.max(best);
            } else {
                beta = beta.min(best);
            }
        }
        if alpha < beta {
            let mut legal = self.take_buf();
            pos.legal_moves_into(&mut legal);
            let mut order = self.take_buf();
            self.ordered(pos, &legal, tt_move, false, &mut order);
            let mut res = Ok(());
            for &m in &order {
                let (d, nf, term) = self.child(pos, m, flags);
                let v = if term {
                    d
                } else {
                    match self.search(pos, depth - 1, alpha - d, beta - d, &nf) {
                        Ok(x) => d + x,
                        Err(e) => {
                            pos.pop().unwrap();
                            res = Err(e);
                            break;
                        }
                    }
                };
                pos.pop().unwrap();
                if maxing {
                    if v > best {
                        best = v;
                        best_move = m;
                        alpha = alpha.max(v);
                    }
                } else if v < best {
                    best = v;
                    best_move = m;
                    beta = beta.min(v);
                }
                if alpha >= beta {
                    break;
                }
            }
            self.buf.push(legal);
            self.buf.push(order);
            res?;
        }
        let flag = if a0 < best && best < b0 { 0 } else if best >= b0 { 1 } else { 2 };
        if self.tt.len() < self.tt_size {
            self.tt.insert(k, Tte { depth, flag, v: best, m: best_move });
        }
        Ok(best)
    }

    /// The move for the side to move (RESIGN = claim the win), as bot/greedy.py `best_move`.
    pub fn best_move(&mut self, pos: &mut Position) -> GreedyResult {
        let t0 = Instant::now();
        self.me = pos.current_player();
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return GreedyResult { best: RESIGN, ..Default::default() };
        }
        let f0 = Flags { endgame: endgame_now(&pos.board), trade_v: -1, trade_plies: 0 };
        self.nodes = 0;
        self.root_material = material(&pos.board, self.me);
        self.deadline = self.time_ms.map(|ms| t0 + std::time::Duration::from_secs_f64(ms / 1000.0));
        if self.tt.len() > self.tt_size / 2 {
            self.tt.clear();
        }
        let claim = self.claim(pos);
        let (mut best, mut bv, mut reached) = (moves[0], -INF, 0);
        let limit = match (self.fixed_depth, self.time_ms.is_some() || self.node_limit.is_some()) {
            (Some(d), _) => d,
            (None, true) => self.max_depth,
            (None, false) => self.p.depth,
        };
        let mut order = Vec::new();
        self.ordered(pos, &moves, NO_MOVE, false, &mut order);
        let root_depth = pos.depth();
        'deepen: for d in 1..=limit {
            let mut alpha = -INF;
            let (mut cur_best, mut cur_v) = (NO_MOVE, -INF);
            let mut scores: Vec<(f64, Move)> = Vec::with_capacity(order.len());
            for &m in &order {
                let (dl, nf, term) = self.child(pos, m, &f0);
                let v = if term {
                    dl
                } else {
                    match self.search(pos, d - 1, alpha - dl, INF, &nf) {
                        Ok(x) => dl + x,
                        Err(_) => {
                            while pos.depth() > root_depth {
                                pos.pop().unwrap();
                            }
                            break 'deepen;
                        }
                    }
                };
                pos.pop().unwrap();
                scores.push((v, m));
                if v > cur_v {
                    cur_v = v;
                    cur_best = m;
                    alpha = alpha.max(v);
                }
            }
            best = cur_best;
            bv = cur_v;
            reached = d;
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            order = scores.iter().map(|x| x.1).collect();
        }
        if let Some(c) = claim {
            if c > bv {
                best = RESIGN;
                bv = c;
            }
        }
        GreedyResult { best, depth: reached, nodes: self.nodes, value: bv, seconds: t0.elapsed().as_secs_f64() }
    }
}
