//! max^n search over the NNUE, a port of `bot/maxn.py` (the Python version stays as the
//! reference; the tests check both pick the same moves at a fixed depth).
//!
//! Every player picks the move that maximizes their own utility: their expected chess.com
//! FFA score (mean over opponents of P(finish above), ties half), from the NNUE at the
//! leaves and from the final scores at finished games. Utilities are in absolute seat
//! order (R, B, Y, G) and sum to 2.
//!
//! - iterative deepening (depth 1, 2, ...) inside a time budget, or to a fixed depth
//! - quiescence on point-scoring captures at the leaves (the mover may also stand pat)
//! - a transposition table keyed by board, side to move, scores and eliminated seats
//! - move ordering: the table's move first, then captures by points, then checks (stable)
//! - check extensions: a move that newly attacks an active king is searched one move deeper,
//!   up to `check_ext` times along a line, so chains of checks are followed; the first
//!   quiescence step also plays out checks (where "predator and prey" play happens)
//! - selective depth (`width`): from ply `full_plies` on, a node searches only the `width`
//!   children with the best static evaluation for the mover (plus the table's move), so deep
//!   searches (depth 5) fit in a self-play budget; 0 = search every move
//! - preferences (both off by default), applied to a child's value as it is backed up:
//!   - `discount` (< 1): every move pulls the values towards the mean 0.5, so a player who is
//!     doing well prefers the shorter line (and one doing badly the longer one)
//!   - `elim_bonus`: a move that eliminates a player adds `elim_bonus * (1 - turn / elim_horizon)`
//!     (not below 0) to the mover's value, so eliminating early is preferred
//!   Both are kept smaller than what the network gives a pawn, so they break near-ties and
//!   never outweigh keeping one's own pieces (logs/nnue_selfplay_plan.md)
//! - immediate pruning: a utility of 1 (above all three) can't be beaten
//! - all sibling leaves are evaluated together in one batched network pass
//! - if the time runs out before depth 1 finishes, the best child by one batched evaluation
//!   is played (depth 0 in the result)
//! - `node_limit` (playground): the search aborts like a timeout once the node count passes
//!   the limit, keeping the last completed depth
//! - `paranoid` (playground, off by default): paranoid alpha-beta. The searching player
//!   maximizes its own utility, every other player is assumed to minimize it. A separate
//!   transposition table holds bounds (the root player is part of its key).
//!
//! The position and the accumulators are made and unmade together, so nothing is copied.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::nnue::{AccStack, Net, NUM_OUTPUTS};
use crate::rules::{splitmix, Move, Position};

pub type Utilities = [f64; 4];

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const WIN: f64 = 1.0 - 1e-9;

/// Utilities (absolute seats) from the 6 pairwise logits of mover k (bot.nnue.utilities).
pub fn utilities_from_logits(logits: &[f32; NUM_OUTPUTS], k: usize) -> Utilities {
    let mut rel = [0.0f64; 4];
    for (n, &(i, j)) in PAIRS.iter().enumerate() {
        let p = 1.0 / (1.0 + (-(logits[n] as f64)).exp());
        rel[i] += p;
        rel[j] += 1.0 - p;
    }
    let mut abs = [0.0f64; 4];
    for i in 0..4 {
        abs[(k + i) % 4] = rel[i] / 3.0;
    }
    abs
}

/// Order of two selection keys (x of one child, y of another): higher first, or lower first
/// when `ascending`.
#[inline]
fn key_order(x: f64, y: f64, ascending: bool) -> std::cmp::Ordering {
    let o = if ascending { x.partial_cmp(&y) } else { y.partial_cmp(&x) };
    o.unwrap()
}

/// Final FFA utilities from the scores (bot.maxn.exact_utilities).
pub fn exact_utilities(pos: &Position) -> Utilities {
    let s = pos.scores;
    let mut u = [0.0f64; 4];
    for i in 0..4 {
        let mut r = 0.0;
        for j in 0..4 {
            if i != j {
                r += if s[i] > s[j] { 1.0 } else if s[i] == s[j] { 0.5 } else { 0.0 };
            }
        }
        u[i] = r / 3.0;
    }
    u
}

#[derive(Clone, Copy)]
struct Entry {
    depth: u32,
    value: Utilities,
    best: Option<Move>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

/// Paranoid table entry: the root player's value and what bound it is.
#[derive(Clone, Copy)]
struct PEntry {
    depth: u32,
    value: f64,
    best: Option<Move>,
    bound: Bound,
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub best: Move,
    pub depth: u32,
    pub nodes: u64,
    pub value: Option<Utilities>,
    pub forced: bool,
    pub seconds: f64,
}

pub struct Searcher {
    pub net: Arc<Net>,
    pub time_ms: Option<f64>,
    pub depth: Option<u32>,
    pub max_depth: u32,
    pub qdepth: u32,
    pub tt_size: usize,
    pub check_ext: u32,
    pub width: usize,
    pub full_plies: u32,
    pub discount: f64,
    pub elim_bonus: f64,
    pub elim_horizon: u32,
    /// Abort once this many nodes have been searched (None = no limit).
    pub node_limit: Option<u64>,
    /// Paranoid alpha-beta instead of max^n (see `search_paranoid`).
    pub paranoid: bool,
    /// Paranoid only: where the selective width applies at nodes that are not the root
    /// player's, keep the children with the LOWEST root utility (the opponents' likely replies).
    pub paranoid_min_select: bool,
    /// Paranoid only: search the kept children in the order of their static evaluation (the
    /// selection key), the table's move first.
    pub paranoid_sort: bool,
    tt: HashMap<u64, Entry>,
    tt_p: HashMap<u64, PEntry>,
    nodes: u64,
    deadline: Option<Instant>,
}

impl Searcher {
    pub fn new(net: Arc<Net>, time_ms: Option<f64>, depth: Option<u32>, max_depth: u32, qdepth: u32,
               tt_size: usize, check_ext: u32, width: usize, full_plies: u32, discount: f64, elim_bonus: f64,
               elim_horizon: u32) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&discount) || discount == 0.0 {
            return Err("discount must be in (0, 1]".into());
        }
        if time_ms.is_none() && depth.is_none() {
            return Err("give a time budget or a fixed depth".into());
        }
        Ok(Searcher { net, time_ms, depth, max_depth, qdepth, tt_size, check_ext, width, full_plies, discount, elim_bonus,
                      elim_horizon: elim_horizon.max(1), node_limit: None, paranoid: false,
                      paranoid_min_select: false, paranoid_sort: false,
                      tt: HashMap::new(), tt_p: HashMap::new(), nodes: 0, deadline: None })
    }

    pub fn set_node_limit(&mut self, limit: Option<u64>) {
        self.node_limit = limit;
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.tt_p.clear();
    }

    pub fn best_move(&mut self, pos: &mut Position) -> Result<SearchResult, String> {
        let t0 = Instant::now();
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return Err("no legal moves".into());
        }
        if moves.len() == 1 {
            return Ok(SearchResult { best: moves[0], forced: true, ..Default::default() });
        }
        self.deadline = self.time_ms.map(|ms| t0 + Duration::from_secs_f64(ms / 1000.0));
        self.nodes = 0;
        if self.tt.len() > self.tt_size {
            self.tt.clear();
        }
        if self.tt_p.len() > self.tt_size {
            self.tt_p.clear();
        }
        let mut acc = AccStack::new(self.net.clone(), &pos.board)?;
        let me = pos.current_player();
        let mut res = SearchResult { best: moves[0], ..Default::default() };
        for d in 1..=self.depth.unwrap_or(self.max_depth) {
            let done = if self.paranoid {
                // paranoid: only the root player's utility is backed up; it is reported in
                // the root seat of `value`, the others are 0
                self.search_paranoid(pos, &mut acc, d, self.check_ext, 0, f64::NEG_INFINITY, f64::INFINITY, me)
                    .map(|(s, m)| {
                        let mut v = [0.0; 4];
                        v[me] = s;
                        (v, m)
                    })
            } else {
                self.search(pos, &mut acc, d, self.check_ext, 0)
            };
            match done {
                None => break, // out of time or nodes: keep the last completed depth
                Some((v, m)) => {
                    res.best = m.unwrap_or(moves[0]);
                    res.value = Some(v);
                    res.depth = d;
                    if v[me] >= WIN {
                        break;
                    }
                }
            }
        }
        if res.value.is_none() {
            // Out of time before depth 1 finished (a large network on a short clock): play
            // the best child by one batched evaluation, without quiescence, not moves[0].
            let kids = self.children(pos, &acc, &moves);
            let i = (0..moves.len()).fold(0, |b, i| if kids[i].1[me] > kids[b].1[me] { i } else { b });
            res.best = moves[i];
            res.value = Some(kids[i].1);
        }
        res.nodes = self.nodes;
        res.seconds = t0.elapsed().as_secs_f64();
        Ok(res)
    }

    /// Count a node; true when the node limit is passed or the time is up (checked every 64
    /// nodes).
    #[inline]
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.over_nodes() {
            return true;
        }
        self.nodes & 63 == 0 && self.deadline.is_some_and(|d| Instant::now() > d)
    }

    /// True once the node count has passed `node_limit` (checked after batched evaluations).
    #[inline]
    fn over_nodes(&self) -> bool {
        self.node_limit.is_some_and(|l| self.nodes > l)
    }

    fn key(pos: &Position) -> u64 {
        let mut h = splitmix(((pos.turn % 4) as u64) | ((pos.eliminated as u64) << 2));
        for s in pos.scores {
            h = splitmix(h ^ (s as u32 as u64));
        }
        pos.hash ^ h
    }

    /// For each move: (gives check, eliminates a player) (make + unmake).
    fn flags(pos: &mut Position, moves: &[Move]) -> Vec<(bool, bool)> {
        moves.iter().map(|&m| {
            let info = pos.push(m).expect("legal move");
            pos.pop().expect("pop");
            (info.gives_check, info.eliminated.is_some())
        }).collect()
    }

    /// A child's value as seen from its parent: the game-length discount, then the
    /// elimination bonus for the mover (`turn`: the parent's turn).
    fn shape(&self, v: Utilities, mover: usize, turn: u32, eliminates: bool) -> Utilities {
        let mut out = v;
        for i in 0..4 {
            out[i] = self.shape_component(v[i], i, mover, turn, eliminates);
        }
        out
    }

    /// The elimination bonus added to the mover's component (0 if no bonus applies).
    #[inline]
    fn elim_term(&self, mover: usize, comp: usize, turn: u32, eliminates: bool) -> f64 {
        if eliminates && comp == mover && self.elim_bonus > 0.0 {
            self.elim_bonus * (1.0 - turn as f64 / self.elim_horizon as f64).max(0.0)
        } else {
            0.0
        }
    }

    /// One component of `shape` (the same arithmetic, so the paranoid search can shape the
    /// root player's scalar alone).
    #[inline]
    fn shape_component(&self, x: f64, comp: usize, mover: usize, turn: u32, eliminates: bool) -> f64 {
        let mut x = x;
        if self.discount < 1.0 {
            x = 0.5 + self.discount * (x - 0.5);
        }
        x + self.elim_term(mover, comp, turn, eliminates)
    }

    /// Inverse of `shape_component`: the value a child must have for the shaped value to be
    /// `y` (used to pass alpha-beta windows down through the shaping).
    #[inline]
    fn unshape_component(&self, y: f64, comp: usize, mover: usize, turn: u32, eliminates: bool) -> f64 {
        let y = y - self.elim_term(mover, comp, turn, eliminates);
        if self.discount < 1.0 {
            0.5 + (y - 0.5) / self.discount
        } else {
            y
        }
    }

    /// Moves in search order with their (gives check, eliminates) flags: the table's move,
    /// then captures by points, then checks (stable, like Python's sort).
    fn ordered(pos: &mut Position, tt_move: Option<Move>) -> Vec<(Move, bool, bool)> {
        let moves = pos.legal_moves();
        let fl = Self::flags(pos, &moves);
        let mut kids: Vec<(Move, bool, bool)> = moves.into_iter().zip(fl).map(|(m, (c, e))| (m, c, e)).collect();
        kids.sort_by_key(|&(m, c, _)| (-pos.capture_points(m), -(c as i32)));
        if let Some(t) = tt_move {
            if let Some(i) = kids.iter().position(|&(m, _, _)| m == t) {
                let e = kids.remove(i);
                kids.insert(0, e);
            }
        }
        kids
    }

    fn evaluate(&self, pos: &Position, acc: &mut AccStack) -> Utilities {
        let k = pos.current_player();
        utilities_from_logits(&acc.logits(k, &pos.globals(k)), k)
    }

    /// Utilities of each child after `moves`: exact for finished games, otherwise from one
    /// batched network pass over all live children. Returns (finished, utilities) per move.
    fn children(&mut self, pos: &mut Position, acc: &AccStack, moves: &[Move]) -> Vec<(bool, Utilities)> {
        let mut out = vec![(false, [0.0; 4]); moves.len()];
        let (mut boards, mut ks, mut globals, mut live) = (vec![], vec![], vec![], vec![]);
        for (i, &m) in moves.iter().enumerate() {
            pos.push(m).expect("legal move");
            if pos.game_over().is_some() {
                out[i] = (true, exact_utilities(pos));
            } else {
                let k = pos.current_player();
                boards.push(pos.board);
                ks.push(k);
                globals.extend_from_slice(&pos.globals(k));
                live.push(i);
            }
            pos.pop().expect("pop");
        }
        if !live.is_empty() {
            self.nodes += live.len() as u64;
            let logits = acc.child_logits(&boards, &ks, &globals).expect("child boards");
            for (n, &i) in live.iter().enumerate() {
                out[i] = (false, utilities_from_logits(&logits[n], ks[n]));
            }
        }
        out
    }

    fn push(pos: &mut Position, acc: &mut AccStack, m: Move) {
        pos.push(m).expect("legal move");
        acc.push(&pos.board).expect("board");
    }

    fn pop(pos: &mut Position, acc: &mut AccStack) {
        acc.pop().expect("acc pop");
        pos.pop().expect("pos pop");
    }

    /// None when the time ran out (the position and accumulators are restored either way).
    /// `ext`: check extensions still allowed on this line.
    /// `ply`: distance from the root (0 = root).
    fn search(&mut self, pos: &mut Position, acc: &mut AccStack, depth: u32, ext: u32, ply: u32)
              -> Option<(Utilities, Option<Move>)> {
        let root = ply == 0;
        if self.tick() {
            return None;
        }
        if pos.game_over().is_some() {
            return Some((exact_utilities(pos), None));
        }
        if depth == 0 {
            let stand = self.evaluate(pos, acc);
            return self.quiesce(pos, acc, self.qdepth, stand).map(|v| (v, None));
        }
        let key = Self::key(pos);
        let hit = self.tt.get(&key).copied();
        if let Some(h) = hit {
            if h.depth >= depth && !root {
                return Some((h.value, h.best));
            }
        }
        let me = pos.current_player();
        let tt_move = hit.and_then(|h| h.best);
        let kids = Self::ordered(pos, tt_move);
        if kids.is_empty() {
            return Some((self.evaluate(pos, acc), None));
        }
        let (kids, leaves) = self.select(pos, acc, depth, ext, ply, tt_move, kids, me, false, false);
        if self.over_nodes() {
            return None;
        }
        let turn = pos.turn;
        let (mut best_v, mut best_m): (Option<Utilities>, Option<Move>) = (None, None);
        for (i, &(m, chk, elim)) in kids.iter().enumerate() {
            let extend = chk && ext > 0;
            if depth == 1 && self.tick() {
                return None;
            }
            let v = if let Some((finished, stand)) = leaves[i] {
                if finished {
                    stand
                } else {
                    Self::push(pos, acc, m);
                    let r = self.quiesce(pos, acc, self.qdepth, stand);
                    Self::pop(pos, acc);
                    r?
                }
            } else {
                Self::push(pos, acc, m);
                let r = if pos.game_over().is_some() {
                    Some((exact_utilities(pos), None))
                } else if extend {
                    self.search(pos, acc, depth, ext - 1, ply + 1)
                } else {
                    self.search(pos, acc, depth - 1, ext, ply + 1)
                };
                Self::pop(pos, acc);
                r?.0
            };
            let sv = self.shape(v, me, turn, elim);
            if best_v.is_none_or(|b| sv[me] > b[me]) {
                best_v = Some(sv);
                best_m = Some(m);
                if v[me] >= WIN {
                    break;          // finishes above everyone (judged before the preferences)
                }
            }
        }
        let best_v = best_v.unwrap();
        self.tt.insert(key, Entry { depth, value: best_v, best: best_m });
        Some((best_v, best_m))
    }

    /// The children to search at a node, in search order, and (at depth 1) their static
    /// evaluations: None = search the child, Some((finished, utilities)) = a leaf.
    /// Selective depth: from ply `full_plies` on, only the `width` children with the best
    /// static evaluation (plus the table's move, which `ordered` put first). "Best" is the
    /// highest `comp` utility, or the lowest when `ascending` (max^n: the mover's own utility,
    /// descending). `sort_kept` puts the children in that same key order (the table's move
    /// first) instead of move order; at depth 1 the children without a leaf (extended checks)
    /// go first, in move order, then the leaves by key.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn select(&mut self, pos: &mut Position, acc: &AccStack, depth: u32, ext: u32, ply: u32,
              tt_move: Option<Move>, kids: Vec<(Move, bool, bool)>, comp: usize, ascending: bool, sort_kept: bool)
              -> (Vec<(Move, bool, bool)>, Vec<Option<(bool, Utilities)>>) {
        let n = kids.len();
        let tt_first = tt_move.is_some_and(|t| kids[0].0 == t);
        let mut leaves: Vec<Option<(bool, Utilities)>> = vec![None; n];
        let mut order: Vec<usize> = (0..n).collect();
        if self.width > 0 && ply >= self.full_plies && n > self.width {
            let ms: Vec<Move> = kids.iter().map(|&(m, _, _)| m).collect();
            let evals = self.children(pos, acc, &ms);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| key_order(evals[a].1[comp], evals[b].1[comp], ascending).then(a.cmp(&b)));
            let mut keep = vec![false; n];
            for &i in idx.iter().take(self.width) {
                keep[i] = true;
            }
            if tt_first {
                keep[0] = true;
            }
            order = if sort_kept {
                // idx is already in key order
                let mut o: Vec<usize> = idx.iter().copied().filter(|&i| keep[i]).collect();
                if tt_first {
                    o.retain(|&i| i != 0);
                    o.insert(0, 0);
                }
                o
            } else {
                (0..n).filter(|&i| keep[i]).collect()
            };
            for &i in &order {
                leaves[i] = if depth == 1 && !(kids[i].1 && ext > 0) { Some(evals[i]) } else { None };
            }
        } else if depth == 1 {
            // depth 1: every child that isn't extended is a leaf, evaluated in one batch
            let idx: Vec<usize> = (0..n).filter(|&i| !(kids[i].1 && ext > 0)).collect();
            let ms: Vec<Move> = idx.iter().map(|&i| kids[i].0).collect();
            for (j, r) in self.children(pos, acc, &ms).into_iter().enumerate() {
                leaves[idx[j]] = Some(r);
            }
            if sort_kept {
                let rest: Vec<usize> = (0..n).filter(|&i| !(tt_first && i == 0)).collect();
                let mut o: Vec<usize> = Vec::with_capacity(n);
                if tt_first {
                    o.push(0);
                }
                o.extend(rest.iter().copied().filter(|&i| leaves[i].is_none()));
                let mut with_leaf: Vec<usize> = rest.iter().copied().filter(|&i| leaves[i].is_some()).collect();
                with_leaf.sort_by(|&a, &b| {
                    let (x, y) = (leaves[a].unwrap().1[comp], leaves[b].unwrap().1[comp]);
                    key_order(x, y, ascending).then(a.cmp(&b))
                });
                o.extend(with_leaf);
                order = o;
            }
        }
        let k2 = order.iter().map(|&i| kids[i]).collect();
        let l2 = order.iter().map(|&i| leaves[i]).collect();
        (k2, l2)
    }

    /// The legal moves of a quiescence node in search order, with their (gives check,
    /// eliminates) flags: captures by points, plus checks on the first step (`first`).
    fn quiesce_moves(pos: &mut Position, first: bool) -> Vec<(Move, bool, bool)> {
        let mut moves = pos.legal_moves();
        if !first {
            moves.retain(|&m| pos.capture_points(m) > 0);
        }
        let fl = Self::flags(pos, &moves);
        let mut kids: Vec<(Move, bool, bool)> = moves.into_iter().zip(fl).map(|(m, (c, e))| (m, c, e))
            .filter(|&(m, c, _)| pos.capture_points(m) > 0 || (first && c)).collect();
        kids.sort_by_key(|&(m, c, _)| (-pos.capture_points(m), -(c as i32)));
        kids
    }

    /// Stand pat (`stand`, this position's evaluation) or play a point-scoring capture; on
    /// the first quiescence step also a check.
    fn quiesce(&mut self, pos: &mut Position, acc: &mut AccStack, q: u32, stand: Utilities) -> Option<Utilities> {
        if q == 0 {
            return Some(stand);
        }
        let me = pos.current_player();
        let first = q == self.qdepth;
        let kids = Self::quiesce_moves(pos, first);
        if kids.is_empty() {
            return Some(stand);
        }
        let ms: Vec<Move> = kids.iter().map(|&(m, _, _)| m).collect();
        let turn = pos.turn;
        let evals = self.children(pos, acc, &ms);
        if self.over_nodes() {
            return None;
        }
        let mut best = stand;
        for (i, &m) in ms.iter().enumerate() {
            if self.tick() {
                return None;
            }
            let (finished, ev) = evals[i];
            let v = if finished || q == 1 {
                ev
            } else {
                Self::push(pos, acc, m);
                let r = self.quiesce(pos, acc, q - 1, ev);
                Self::pop(pos, acc);
                r?
            };
            let sv = self.shape(v, me, turn, kids[i].2);
            if sv[me] > best[me] {
                best = sv;
            }
        }
        Some(best)
    }

    /// Paranoid alpha-beta over the root player's utility (a scalar). At nodes where the
    /// side to move is `root` the value is maximized, at every other node minimized.
    /// `alpha`/`beta` are this node's window in its own value space (the space its parent
    /// compares after shaping); the children's windows are the inverse-shaped ones.
    /// Returns (value, best move). None when the time or the node limit ran out.
    #[allow(clippy::too_many_arguments)]
    fn search_paranoid(&mut self, pos: &mut Position, acc: &mut AccStack, depth: u32, ext: u32, ply: u32,
                       alpha: f64, beta: f64, root: usize) -> Option<(f64, Option<Move>)> {
        if self.tick() {
            return None;
        }
        if pos.game_over().is_some() {
            return Some((exact_utilities(pos)[root], None));
        }
        if depth == 0 {
            let stand = self.evaluate(pos, acc)[root];
            return self.quiesce_paranoid(pos, acc, self.qdepth, stand, root, alpha, beta).map(|v| (v, None));
        }
        let key = Self::key(pos) ^ splitmix(0x5041_5241 ^ root as u64);
        let hit = self.tt_p.get(&key).copied();
        if let Some(h) = hit {
            if h.depth >= depth && ply > 0 {
                let cut = match h.bound {
                    Bound::Exact => true,
                    Bound::Lower => h.value >= beta,
                    Bound::Upper => h.value <= alpha,
                };
                if cut {
                    return Some((h.value, h.best));
                }
            }
        }
        let me = pos.current_player();
        let maximizing = me == root;
        let turn = pos.turn;
        let tt_move = hit.and_then(|h| h.best);
        let kids = Self::ordered(pos, tt_move);
        if kids.is_empty() {
            return Some((self.evaluate(pos, acc)[root], None));
        }
        // selection and order key: the root player's utility at the opponents' nodes when
        // paranoid_min_select (lowest first), else the side to move's own utility (highest first)
        let (comp, ascending) = if self.paranoid_min_select && me != root { (root, true) } else { (me, false) };
        let (kids, leaves) = self.select(pos, acc, depth, ext, ply, tt_move, kids, comp, ascending,
                                         self.paranoid_sort);
        if self.over_nodes() {
            return None;
        }
        let (mut a, mut b) = (alpha, beta);
        let mut best: Option<(f64, Option<Move>)> = None;
        for (i, &(m, chk, elim)) in kids.iter().enumerate() {
            let extend = chk && ext > 0;
            if depth == 1 && self.tick() {
                return None;
            }
            let (ca, cb) = (self.unshape_component(a, root, me, turn, elim),
                            self.unshape_component(b, root, me, turn, elim));
            let v = if let Some((finished, stand)) = leaves[i] {
                if finished {
                    stand[root]
                } else {
                    Self::push(pos, acc, m);
                    let r = self.quiesce_paranoid(pos, acc, self.qdepth, stand[root], root, ca, cb);
                    Self::pop(pos, acc);
                    r?
                }
            } else {
                Self::push(pos, acc, m);
                let r = if pos.game_over().is_some() {
                    Some((exact_utilities(pos)[root], None))
                } else if extend {
                    self.search_paranoid(pos, acc, depth, ext - 1, ply + 1, ca, cb, root)
                } else {
                    self.search_paranoid(pos, acc, depth - 1, ext, ply + 1, ca, cb, root)
                };
                Self::pop(pos, acc);
                r?.0
            };
            let sv = self.shape_component(v, root, me, turn, elim);
            let better = match best {
                None => true,
                Some((bv, _)) => if maximizing { sv > bv } else { sv < bv },
            };
            if better {
                best = Some((sv, Some(m)));
            }
            let bv = best.unwrap().0;
            if maximizing {
                a = a.max(bv);
                // a utility of 1 (above all three) can't be beaten
                if v >= WIN || bv >= b {
                    break;
                }
            } else {
                b = b.min(bv);
                if bv <= a {
                    break;
                }
            }
        }
        let (bv, bm) = best.unwrap();
        let bound = if bv <= alpha {
            Bound::Upper
        } else if bv >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_p.insert(key, PEntry { depth, value: bv, best: bm, bound });
        Some((bv, bm))
    }

    /// Quiescence for `search_paranoid`: the scalar is the root player's; the side to move
    /// may stand pat (`stand`, this position's evaluation for the root player). Same move
    /// set as `quiesce`.
    #[allow(clippy::too_many_arguments)]
    fn quiesce_paranoid(&mut self, pos: &mut Position, acc: &mut AccStack, q: u32, stand: f64, root: usize,
                        alpha: f64, beta: f64) -> Option<f64> {
        if q == 0 {
            return Some(stand);
        }
        let me = pos.current_player();
        let maximizing = me == root;
        let kids = Self::quiesce_moves(pos, q == self.qdepth);
        if kids.is_empty() {
            return Some(stand);
        }
        let ms: Vec<Move> = kids.iter().map(|&(m, _, _)| m).collect();
        let turn = pos.turn;
        let evals = self.children(pos, acc, &ms);
        if self.over_nodes() {
            return None;
        }
        let (mut a, mut b) = (alpha, beta);
        let mut best = stand;
        if maximizing {
            a = a.max(best);
            if best >= b {
                return Some(best);
            }
        } else {
            b = b.min(best);
            if best <= a {
                return Some(best);
            }
        }
        for (i, &m) in ms.iter().enumerate() {
            if self.tick() {
                return None;
            }
            let (finished, ev) = evals[i];
            let elim = kids[i].2;
            let (ca, cb) = (self.unshape_component(a, root, me, turn, elim),
                            self.unshape_component(b, root, me, turn, elim));
            let v = if finished || q == 1 {
                ev[root]
            } else {
                Self::push(pos, acc, m);
                let r = self.quiesce_paranoid(pos, acc, q - 1, ev[root], root, ca, cb);
                Self::pop(pos, acc);
                r?
            };
            let sv = self.shape_component(v, root, me, turn, elim);
            if maximizing {
                if sv > best {
                    best = sv;
                }
                a = a.max(best);
                if best >= b {
                    break;
                }
            } else {
                if sv < best {
                    best = sv;
                }
                b = b.min(best);
                if best <= a {
                    break;
                }
            }
        }
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::{Layer, NUM_FEATURES, NUM_GLOBALS};

    fn tiny_net() -> Arc<Net> {
        let mut s = 3u64;
        let mut v = |n: usize, scale: f32| {
            (0..n).map(|_| { s = splitmix(s); ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale }).collect::<Vec<f32>>()
        };
        let h = 8;
        let w0 = 4 * h + NUM_GLOBALS;
        let l1 = Layer::new(v(16 * w0, 0.5), v(16, 0.5), w0, 16, false).unwrap();
        let out = Layer::new(v(6 * 16, 1.0), v(6, 0.2), 16, 6, false).unwrap();
        Arc::new(Net::new(h, v(NUM_FEATURES * h, 0.2), v(h, 0.2), vec![l1], out).unwrap())
    }

    #[test]
    fn search_restores_the_position_and_is_legal() {
        let mut pos = Position::initial();
        for _ in 0..12 {
            let m = pos.legal_moves()[0];
            pos.push(m).unwrap();
        }
        let before = (pos.board, pos.turn, pos.scores, pos.depth());
        let mut s = Searcher::new(tiny_net(), None, Some(3), 12, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
        let r = s.best_move(&mut pos).unwrap();
        assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before);
        assert!(pos.legal_moves().contains(&r.best));
        assert_eq!(r.depth, 3);
        let u = r.value.unwrap();
        assert!((u.iter().sum::<f64>() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn time_budget_is_respected() {
        let mut pos = Position::initial();
        let mut s = Searcher::new(tiny_net(), Some(50.0), None, 30, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
        let r = s.best_move(&mut pos).unwrap();
        assert!(r.seconds < 0.2, "{}", r.seconds);
        assert!(r.depth >= 1);
        assert_eq!(pos.depth(), 0);
    }

    #[test]
    fn exact_utilities_sum_to_two() {
        let mut pos = Position::initial();
        pos.scores = [10, 3, 10, 0];
        let u = exact_utilities(&pos);
        assert!((u[0] - 2.5 / 3.0).abs() < 1e-12 && u[3] == 0.0);
        assert!((u.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    }

    fn midgame() -> Position {
        let mut pos = Position::initial();
        for _ in 0..12 {
            let m = pos.legal_moves()[0];
            pos.push(m).unwrap();
        }
        pos
    }

    #[test]
    fn paranoid_search_returns_a_legal_move_and_restores_the_position() {
        let mut pos = midgame();
        let before = (pos.board, pos.turn, pos.scores, pos.depth());
        let mut s = Searcher::new(tiny_net(), None, Some(3), 12, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
        s.paranoid = true;
        let r = s.best_move(&mut pos).unwrap();
        assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before);
        assert!(pos.legal_moves().contains(&r.best));
        assert_eq!(r.depth, 3);
        // with the default preferences, paranoid play also works with the shaping on
        let mut s2 = Searcher::new(tiny_net(), None, Some(2), 12, 2, 100_000, 3, 4, 1, 0.9985, 0.008, 200).unwrap();
        s2.paranoid = true;
        let r2 = s2.best_move(&mut pos).unwrap();
        assert!(pos.legal_moves().contains(&r2.best));
        assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before);
    }

    #[test]
    fn paranoid_root_value_is_a_utility() {
        let mut pos = midgame();
        let mut s = Searcher::new(tiny_net(), None, Some(2), 12, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
        s.paranoid = true;
        let r = s.best_move(&mut pos).unwrap();
        let me = pos.current_player();
        let u = r.value.unwrap()[me];
        assert!((0.0..=1.0).contains(&u), "{}", u);
    }

    #[test]
    fn node_limit_is_respected() {
        for paranoid in [false, true] {
            let mut pos = midgame();
            let before = (pos.board, pos.turn, pos.scores, pos.depth());
            let mut s = Searcher::new(tiny_net(), Some(1e9), None, 30, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
            s.paranoid = paranoid;
            s.node_limit = Some(3000);
            let r = s.best_move(&mut pos).unwrap();
            assert!(r.nodes <= 3000 + 500, "paranoid {} nodes {}", paranoid, r.nodes);
            assert!(r.depth >= 1 && r.depth < 30, "depth {}", r.depth);
            assert!(pos.legal_moves().contains(&r.best));
            assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before);
        }
    }

    #[test]
    fn paranoid_selection_options_return_legal_moves_and_restore_the_position() {
        for (pmin, psort) in [(true, false), (false, true), (true, true)] {
            let mut pos = midgame();
            let before = (pos.board, pos.turn, pos.scores, pos.depth());
            let mut s = Searcher::new(tiny_net(), None, Some(3), 12, 2, 100_000, 3, 4, 1, 1.0, 0.0, 200).unwrap();
            s.paranoid = true;
            s.paranoid_min_select = pmin;
            s.paranoid_sort = psort;
            let r = s.best_move(&mut pos).unwrap();
            assert_eq!((pos.board, pos.turn, pos.scores, pos.depth()), before, "pmin {} psort {}", pmin, psort);
            assert!(pos.legal_moves().contains(&r.best), "pmin {} psort {}", pmin, psort);
            assert_eq!(r.depth, 3);
        }
    }

    #[test]
    fn paranoid_sort_changes_only_the_order() {
        // alpha-beta gives the same minimax value whatever the order (the move may differ on ties)
        let mut values = vec![];
        for psort in [false, true] {
            let mut pos = midgame();
            let me = pos.current_player();
            let mut s = Searcher::new(tiny_net(), None, Some(2), 12, 2, 100_000, 3, 0, 2, 1.0, 0.0, 200).unwrap();
            s.paranoid = true;
            s.paranoid_sort = psort;
            let r = s.best_move(&mut pos).unwrap();
            values.push(r.value.unwrap()[me]);
        }
        assert!((values[0] - values[1]).abs() < 1e-12, "{:?}", values);
    }

    #[test]
    fn maxn_ignores_the_paranoid_options() {
        let mut pos = midgame();
        let search = |flags: bool, pos: &mut Position| {
            let mut s = Searcher::new(tiny_net(), None, Some(3), 12, 2, 100_000, 3, 4, 1, 1.0, 0.0, 200).unwrap();
            s.paranoid_min_select = flags;
            s.paranoid_sort = flags;
            s.best_move(pos).unwrap()
        };
        let off = search(false, &mut pos);
        let on = search(true, &mut pos);
        assert_eq!(off.best, on.best);
        assert_eq!(off.value, on.value);
        assert_eq!(off.nodes, on.nodes);
    }
}
